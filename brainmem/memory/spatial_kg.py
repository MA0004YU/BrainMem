"""Spatial Knowledge Graph of Episodic Memory M_epi (Section III-B-2).

The Spatial KG records two types of relations across all rooms visited:

    (room --contains--> object_i)
    (object_i --r--> object_j)

where r is a spatial predicate covering support and containment relations
(on, in, near) and directional or alignment relations (left of, right of,
in front of, behind, above, below, aligned with). Each relation carries a
support count = the number of episodes that confirmed it, which separates
stable layout knowledge from transient observations. Relations confirmed by at
least ``stable_support`` episodes are promoted to "stable" at the end of an
episode. The graph starts empty and grows over time.

Retrieval returns the relations of the current room, ranked by support count
(relations observed in the current scene first).
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set

from brainmem.types import Relation
from brainmem.utils import atomic_write_json, load_json


def _key(*parts: str) -> str:
    return "||".join(parts)


class SpatialKG:
    FILE = "spatial_kg.json"

    def __init__(self, memory_dir: str, stable_support: int = 2):
        self.path = os.path.join(memory_dir, self.FILE)
        self.stable_support = stable_support
        data = load_json(self.path, {})
        # relation key -> {room, subject, predicate, object, support, scenes, stable}
        self.relations: Dict[str, Dict[str, Any]] = data.get("relations", {})
        self._build_indexes()
        self._episode_seen: Set[str] = set()
        self._episode_id = None

    def _build_indexes(self) -> None:
        self.relations_by_room: Dict[str, List[str]] = defaultdict(list)
        for k, r in self.relations.items():
            self.relations_by_room[r["room"]].append(k)

    # ------------------------------------------------------------------ updates
    def begin_episode(self, episode_id: str) -> None:
        """Support counts are incremented at most once per episode (incl. its retries)."""
        if episode_id != self._episode_id:
            self._episode_id = episode_id
            self._episode_seen = set()

    def _confirm(self, room: str, scene: str, subj: str, pred: str, obj: str) -> None:
        k = _key(room, subj, pred, obj)
        r = self.relations.get(k)
        if r is None:
            r = {"room": room, "subject": subj, "predicate": pred, "object": obj,
                 "support": 0, "scenes": [], "stable": False}
            self.relations[k] = r
            self.relations_by_room[room].append(k)
        if k not in self._episode_seen:
            self._episode_seen.add(k)
            r["support"] += 1
        if scene and scene not in r["scenes"]:
            r["scenes"].append(scene)

    def observe(self, room: str, scene: str, objects: Iterable[str], relations: Iterable[Relation]) -> None:
        """Step-level update: add new nodes/edges and confirm observed relations."""
        for o in objects:
            if o:
                self._confirm(room, scene, room, "contains", o)
        for subj, pred, obj in relations:
            if subj and obj and subj != obj:
                self._confirm(room, scene, subj, pred, obj)

    def consolidate(self) -> None:
        """Episode-level update: promote frequently confirmed relations to stable."""
        for k in self._episode_seen:
            r = self.relations[k]
            if r["support"] >= self.stable_support:
                r["stable"] = True

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, room: str, scene: str, n: int) -> Dict[str, Any]:
        keys = self.relations_by_room.get(room, [])
        contains, rels = [], []
        for k in keys:
            r = self.relations[k]
            (contains if r["predicate"] == "contains" else rels).append(r)

        def rank(r: Dict[str, Any]):
            return (scene in r["scenes"], r["support"])

        contains.sort(key=rank, reverse=True)
        rels.sort(key=rank, reverse=True)
        return {"room": room, "contains": contains[: max(n, 1) * 2], "relations": rels[:n]}

    # ------------------------------------------------------------------ persistence / stats
    def save(self) -> None:
        atomic_write_json(self.path, {"relations": self.relations})

    def stats(self) -> Dict[str, int]:
        n_contains = sum(1 for r in self.relations.values() if r["predicate"] == "contains")
        return {"spatial_relations": len(self.relations) - n_contains,
                "spatial_contains": n_contains,
                "spatial_stable": sum(1 for r in self.relations.values() if r["stable"]),
                "spatial_rooms": len(self.relations_by_room)}
