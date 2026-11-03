"""Semantic Memory M_sem (Sections III-B-3 and III-D).

M_sem stores two types of entries.

Experience records -- one per episode, produced by the experience agent:
task instruction, room context, success/failure status, a summarized action
sequence, key failure reasons (if applicable), a key success pattern (if
applicable) and learning insights. Records are indexed by task type, so
retrieval is keyed rather than scanned.

Guidelines -- reusable rules distilled from compatible experience records.
Each guideline holds a natural-language description, applicable task types,
object categories, and the statistics N_success(g), N_total(g):

    confidence(g) = N_success(g) / N_total(g)
    usage(g)      = min(N_success(g) / 10, 1.0)
    utility(g)    = alpha * confidence(g) + beta * usage(g),   alpha + beta = 1

With alpha = 0.7 a guideline that succeeded in 8 of 10 applications
(utility 0.8 for every alpha) outranks a guideline with a single successful
application (utility 0.9 * alpha + 0.1); the two cross at alpha = 7/9 ~ 0.78.

Low-utility pruning: at most M active guidelines are retained. When the
number of guidelines exceeds 1.5 M, they are sorted by utility and the
lowest-scoring ones are removed until M remain. Guidelines with
confidence >= 0.8 and N_success >= 5 are protected from pruning.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from brainmem.utils import atomic_write_json, jaccard, load_json, short_hash, tokens

GENERAL_TASK_TYPE = "general"


def confidence(n_success: int, n_total: int) -> float:
    return n_success / n_total if n_total > 0 else 0.0


def usage(n_success: int, saturation: int = 10) -> float:
    return min(n_success / saturation, 1.0)


def utility(n_success: int, n_total: int, alpha: float, beta: float, saturation: int = 10) -> float:
    return alpha * confidence(n_success, n_total) + beta * usage(n_success, saturation)


class SemanticMemory:
    FILE = "semantic_memory.json"

    def __init__(self, memory_dir: str, alpha: float = 0.7, beta: float = 0.3, usage_saturation: int = 10,
                 max_guidelines: int = 20, prune_trigger: int = 30, protect_confidence: float = 0.8,
                 protect_min_success: int = 5, guideline_merge_threshold: float = 0.6):
        self.path = os.path.join(memory_dir, self.FILE)
        self.alpha, self.beta = alpha, beta
        self.usage_saturation = usage_saturation
        self.max_guidelines = max_guidelines
        self.prune_trigger = prune_trigger
        self.protect_confidence = protect_confidence
        self.protect_min_success = protect_min_success
        self.guideline_merge_threshold = guideline_merge_threshold
        data = load_json(self.path, {})
        self.records: List[Dict[str, Any]] = data.get("records", [])
        self.guidelines: Dict[str, Dict[str, Any]] = data.get("guidelines", {})
        self.counters: Dict[str, int] = data.get("counters", {"created": 0, "merged": 0, "pruned": 0})
        self.records_by_task_type: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(self.records):
            self.records_by_task_type[r["task_type"]].append(i)

    # ------------------------------------------------------------------ scores
    def confidence(self, g: Dict[str, Any]) -> float:
        return confidence(g["n_success"], g["n_total"])

    def utility(self, g: Dict[str, Any]) -> float:
        return utility(g["n_success"], g["n_total"], self.alpha, self.beta, self.usage_saturation)

    def is_protected(self, g: Dict[str, Any]) -> bool:
        return self.confidence(g) >= self.protect_confidence and g["n_success"] >= self.protect_min_success

    def _rescore(self, g: Dict[str, Any]) -> None:
        g["confidence"] = self.confidence(g)
        g["utility"] = self.utility(g)

    # ------------------------------------------------------------------ experience records
    def add_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(record)
        record.setdefault("timestamp", time.time())
        record["record_id"] = f"exp_{len(self.records):06d}"
        self.records.append(record)
        self.records_by_task_type[record["task_type"]].append(len(self.records) - 1)
        return record

    def retrieve_records(self, task_type: str, target_object: str, room_id: str, k: int,
                         exclude_episode: Optional[str] = None) -> List[Dict[str, Any]]:
        cands = [self.records[i] for i in self.records_by_task_type.get(task_type, [])]
        if exclude_episode is not None:
            cands = [r for r in cands if r.get("episode_key") != exclude_episode]
        cands.sort(key=lambda r: (r.get("target_object") == target_object, r.get("room_id") == room_id,
                                  r["timestamp"]), reverse=True)
        return cands[:k]

    # ------------------------------------------------------------------ guidelines
    def add_or_merge_guideline(self, text: str, task_types: Sequence[str], objects: Sequence[str],
                               source_record: str) -> Dict[str, Any]:
        """Insert a distilled guideline, merging it into an equivalent existing one.

        A new guideline originates from a successful trajectory and therefore
        starts with one successful application (N_success = N_total = 1).
        """
        text = text.strip()
        task_types = sorted({t for t in task_types if t}) or [GENERAL_TASK_TYPE]
        objects = sorted({o for o in objects if o})
        toks = tokens(text)
        for g in self.guidelines.values():
            if set(g["task_types"]) & set(task_types) and jaccard(toks, tokens(g["text"])) >= self.guideline_merge_threshold:
                g["task_types"] = sorted(set(g["task_types"]) | set(task_types))
                g["objects"] = sorted(set(g["objects"]) | set(objects))
                if source_record not in g["source_records"]:
                    g["source_records"].append(source_record)
                self.counters["merged"] += 1
                return g
        gid = f"g_{short_hash([text, task_types])}"
        g = {"guideline_id": gid, "text": text, "task_types": task_types, "objects": objects,
             "n_success": 1, "n_total": 1, "created": time.time(), "last_used": None,
             "source_records": [source_record]}
        self._rescore(g)
        self.guidelines[gid] = g
        self.counters["created"] += 1
        return g

    def record_application(self, guideline_ids: Sequence[str], success: bool) -> None:
        """After an episode in which the guidelines were applied, update the counts."""
        now = time.time()
        for gid in set(guideline_ids):
            g = self.guidelines.get(gid)
            if g is None:
                continue
            g["n_total"] += 1
            g["n_success"] += int(bool(success))
            g["last_used"] = now
            self._rescore(g)

    def prune(self) -> List[str]:
        """Low-utility pruning, triggered above the soft limit 1.5 M."""
        if len(self.guidelines) <= self.prune_trigger:
            return []
        for g in self.guidelines.values():
            self._rescore(g)
        protected = [g for g in self.guidelines.values() if self.is_protected(g)]
        others = sorted((g for g in self.guidelines.values() if not self.is_protected(g)),
                        key=lambda g: (g["utility"], g["created"]), reverse=True)
        keep = max(self.max_guidelines - len(protected), 0)
        removed = [g["guideline_id"] for g in others[keep:]]
        for gid in removed:
            del self.guidelines[gid]
        self.counters["pruned"] += len(removed)
        return removed

    def retrieve_guidelines(self, task_type: str, target_object: str, k: int) -> List[Dict[str, Any]]:
        """Top-k guidelines matching the task type and object category, by utility."""
        def rank(g):
            return (target_object in g["objects"], g["utility"], g["created"])

        typed = sorted((g for g in self.guidelines.values() if task_type in g["task_types"]), key=rank, reverse=True)
        if len(typed) < k:
            general = sorted((g for g in self.guidelines.values()
                              if GENERAL_TASK_TYPE in g["task_types"] and task_type not in g["task_types"]),
                             key=rank, reverse=True)
            typed += general[: k - len(typed)]
        return typed[:k]

    # ------------------------------------------------------------------ persistence / stats
    def save(self) -> None:
        atomic_write_json(self.path, {"records": self.records, "guidelines": self.guidelines,
                                      "counters": self.counters})

    def stats(self) -> Dict[str, int]:
        return {"experience_records": len(self.records), "guidelines": len(self.guidelines),
                "guidelines_created": self.counters["created"], "guidelines_merged": self.counters["merged"],
                "guidelines_pruned": self.counters["pruned"]}
