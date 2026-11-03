"""Trajectory Knowledge Graph of Episodic Memory M_epi (Section III-B-2).

State node      s_i     = (state_id, room_id, summary)
Transition edge e_{i->j} = (prev_state_id, next_state_id, action, success, ...)

Construction (per step): observe and summarize the environment, create or
reuse a state node s_t, record the executed action a_t with its outcome, and
add the transition edge e_{t-1 -> t}.

Merging rules:
  * State deduplication  -- states in the same room whose summaries are similar
    (token Jaccard >= ``state_merge_threshold``) reuse the same node id.
  * Transition aggregation -- transitions with identical
    (prev_state, action, next_state) accumulate counts and success counts and
    attach up to ``max_reasoning_examples`` reasoning examples.
  * Failure tagging -- transitions whose action failed are tagged and are
    recalled when a task of the same type is attempted again.

Capacity: the graph is bounded structurally. Revisited situations update
existing nodes and edges; nothing is deleted, because failure-tagged
transitions are negative evidence. At retrieval time candidates are ranked by
success count so that only top-ranked entries enter the prompt.

All lookups go through the indexes ``states_by_room``, ``edges_by_src`` and
``rollouts_by_task_type``, so retrieval cost depends on the number of matching
entries rather than on the number of consolidated episodes.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from brainmem.utils import atomic_write_json, jaccard, load_json, short_hash, tokens


class TrajectoryKG:
    FILE = "trajectory_kg.json"

    def __init__(self, memory_dir: str, state_merge_threshold: float = 0.8,
                 task_similarity_threshold: float = 0.5, max_reasoning_examples: int = 3):
        self.path = os.path.join(memory_dir, self.FILE)
        self.state_merge_threshold = state_merge_threshold
        self.task_similarity_threshold = task_similarity_threshold
        self.max_reasoning_examples = max_reasoning_examples
        data = load_json(self.path, {})
        self.states: Dict[str, Dict[str, Any]] = data.get("states", {})
        self.edges: Dict[str, Dict[str, Any]] = data.get("edges", {})
        self.rollouts: List[Dict[str, Any]] = data.get("rollouts", [])
        self._build_indexes()
        self._active: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ indexes
    def _build_indexes(self) -> None:
        self.states_by_room: Dict[str, List[str]] = defaultdict(list)
        for sid, s in self.states.items():
            self.states_by_room[s["room_id"]].append(sid)
        self.edges_by_src: Dict[str, List[str]] = defaultdict(list)
        for key, e in self.edges.items():
            self.edges_by_src[e["prev_state_id"]].append(key)
        self.rollouts_by_task_type: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(self.rollouts):
            self.rollouts_by_task_type[r["task_type"]].append(i)

    # ------------------------------------------------------------------ states
    def _match_or_create_state(self, room_id: str, summary: str, state_tokens: Sequence[str]) -> str:
        """State deduplication rule."""
        toks = sorted(set(state_tokens))
        best_id, best_sim = None, -1.0
        for sid in self.states_by_room.get(room_id, []):
            sim = jaccard(toks, self.states[sid]["tokens"])
            if sim > best_sim:
                best_id, best_sim = sid, sim
        if best_id is not None and best_sim >= self.state_merge_threshold:
            self.states[best_id]["visits"] += 1
            return best_id
        sid = f"s_{short_hash([room_id, toks])}"
        while sid in self.states:  # distinct token sets that collide on the short hash
            sid = f"s_{short_hash([room_id, toks, len(self.states)])}"
        self.states[sid] = {"state_id": sid, "room_id": room_id, "summary": summary,
                            "tokens": toks, "visits": 1}
        self.states_by_room[room_id].append(sid)
        return sid

    # ------------------------------------------------------------------ rollout lifecycle
    def start_rollout(self, episode_key: str, attempt: int, task_type: str, target_object: str,
                      instruction: str, room_id: str, initial_summary: str,
                      initial_tokens: Sequence[str]) -> str:
        """Create (or reuse) s_0, summarizing the initial scene and the task instruction."""
        summary = f"Task: {instruction} | {initial_summary}"
        toks = list(initial_tokens) + [f"task:{t}" for t in sorted(tokens(instruction))]
        s0 = self._match_or_create_state(room_id, summary, toks)
        self._active = {
            "episode_key": episode_key, "attempt": attempt, "task_type": task_type,
            "target_object": target_object, "instruction_tokens": sorted(tokens(instruction)),
            "room_id": room_id, "edge_keys": [], "actions": [], "current_state": s0,
            "start_state": s0,
        }
        return s0

    @property
    def current_state(self) -> Optional[str]:
        return self._active["current_state"] if self._active else None

    def add_step(self, action: str, success: Optional[bool], room_id: str, summary: str,
                 state_tokens: Sequence[str], reasoning: str = "", feedback: str = "") -> str:
        """Step-level update: create/reuse s_t and add e_{t-1->t}.

        ``success`` is None when execution feedback is not recorded; the edge is
        then stored without outcome statistics.
        """
        assert self._active is not None, "start_rollout() must be called first"
        prev = self._active["current_state"]
        nxt = self._match_or_create_state(room_id, summary, state_tokens)
        key = short_hash([prev, action, nxt], n=16)
        e = self.edges.get(key)
        if e is None:
            e = {"prev_state_id": prev, "next_state_id": nxt, "action": action,
                 "count": 0, "success_count": 0, "failure_count": 0, "failed": False,
                 "goal_success_count": 0, "task_types": {}, "reasoning_examples": [],
                 "failure_feedback": []}
            self.edges[key] = e
            self.edges_by_src[prev].append(key)
        # Transition aggregation
        e["count"] += 1
        tt = e["task_types"].setdefault(self._active["task_type"], {"count": 0, "success": 0, "failure": 0})
        tt["count"] += 1
        if success is True:
            e["success_count"] += 1
            tt["success"] += 1
        elif success is False:
            # Failure tagging
            e["failure_count"] += 1
            e["failed"] = True
            tt["failure"] += 1
            if feedback and feedback not in e["failure_feedback"] and len(e["failure_feedback"]) < self.max_reasoning_examples:
                e["failure_feedback"].append(feedback)
        if reasoning and len(e["reasoning_examples"]) < self.max_reasoning_examples:
            e["reasoning_examples"].append(reasoning[:500])
        self._active["edge_keys"].append(key)
        self._active["actions"].append(action)
        self._active["current_state"] = nxt
        return nxt

    def end_rollout(self, success: Optional[bool]) -> None:
        """Episode-level update: finalize the rollout with the task outcome."""
        if self._active is None:
            return
        r = dict(self._active)
        r["success"] = success
        r["timestamp"] = time.time()
        r.pop("current_state", None)
        if success:
            for key in set(r["edge_keys"]):
                self.edges[key]["goal_success_count"] += 1
        self.rollouts.append(r)
        self.rollouts_by_task_type[r["task_type"]].append(len(self.rollouts) - 1)
        self._active = None

    # ------------------------------------------------------------------ retrieval
    def _similar(self, r: Dict[str, Any], task_type: str, target_object: str, instr_tokens: Sequence[str]) -> bool:
        if r["task_type"] != task_type:
            return False
        if target_object and r["target_object"] == target_object:
            return True
        return jaccard(instr_tokens, r["instruction_tokens"]) >= self.task_similarity_threshold

    def recall_sequences(self, task_type: str, target_object: str, instruction: str,
                         n: int, use_outcome: bool = True) -> List[Tuple[List[str], int]]:
        """Action sequences of highly similar past tasks, ranked by success count.

        With ``use_outcome`` the candidates are the successful rollouts and the
        rank is the number of successful rollouts that share the sequence.
        Without outcome information every similar rollout counts.
        """
        instr_tokens = sorted(tokens(instruction))
        groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for i in self.rollouts_by_task_type.get(task_type, []):
            r = self.rollouts[i]
            if not r["actions"] or not self._similar(r, task_type, target_object, instr_tokens):
                continue
            if use_outcome and not r["success"]:
                continue
            g = groups.setdefault(tuple(r["actions"]), {"count": 0, "last": 0.0})
            g["count"] += 1
            g["last"] = max(g["last"], r["timestamp"])
        ranked = sorted(groups.items(), key=lambda kv: (kv[1]["count"], kv[1]["last"]), reverse=True)
        return [(list(seq), g["count"]) for seq, g in ranked[:n]]

    def failed_transitions(self, task_type: str, room_id: str, current_state: Optional[str],
                           n: int) -> List[Dict[str, Any]]:
        """Failure-tagged transitions recalled for a task of the same type.

        Transitions leaving the current state come first, then those leaving
        other states of the same room; ties are broken by failure count.
        """
        def collect(state_ids: Sequence[str], from_current: bool) -> List[Tuple[Tuple[int, int], Dict[str, Any]]]:
            out = []
            for sid in state_ids:
                for key in self.edges_by_src.get(sid, []):
                    e = self.edges[key]
                    stats = e["task_types"].get(task_type)
                    if not e["failed"] or not stats or stats["failure"] == 0:
                        continue
                    out.append(((int(from_current), stats["failure"]), e))
            return out

        cands = collect([current_state] if current_state else [], True)
        cands += collect([s for s in self.states_by_room.get(room_id, []) if s != current_state], False)
        cands.sort(key=lambda c: c[0], reverse=True)
        seen, result = set(), []
        for _, e in cands:
            if e["action"] in seen:
                continue
            seen.add(e["action"])
            result.append({"action": e["action"], "failures": e["task_types"][task_type]["failure"],
                           "feedback": e["failure_feedback"][0] if e["failure_feedback"] else "",
                           "state_summary": self.states[e["prev_state_id"]]["summary"]})
            if len(result) >= n:
                break
        return result

    # ------------------------------------------------------------------ persistence / stats
    def save(self) -> None:
        atomic_write_json(self.path, {"states": self.states, "edges": self.edges, "rollouts": self.rollouts})

    def stats(self) -> Dict[str, int]:
        return {"trajectory_nodes": len(self.states), "trajectory_edges": len(self.edges),
                "trajectory_failure_tagged_edges": sum(1 for e in self.edges.values() if e["failed"]),
                "trajectory_rollouts": len(self.rollouts)}
