"""Data types exchanged between the benchmark adapters and the memory system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# (subject, predicate, object), e.g. ("fork", "on", "countertop")
Relation = Tuple[str, str, str]


@dataclass
class TaskQuery:
    """Task query Q = {room, task_type, object} of Section III-C."""
    instruction: str
    task_type: str
    target_object: str
    room_id: str
    scene_id: str
    episode_key: str  # stable identifier of the benchmark episode


@dataclass
class StepObservation:
    """What the adapter extracts from the simulator after reset or after a step."""
    room_id: str
    scene_id: str
    summary: str                     # short text: visible objects, agent pose, key relations
    state_tokens: List[str]          # tokens used for state deduplication
    agent_state: str                 # e.g. "holding fork" / "holding nothing"
    location: str                    # e.g. "kitchen (near CounterTop)"
    objects: List[str] = field(default_factory=list)       # objects observed in this room
    relations: List[Relation] = field(default_factory=list)  # object-object relations observed


@dataclass
class StepRecord:
    """One executed environment step of a rollout."""
    t: int
    action: str
    feedback: str
    success: bool
    reasoning: str = ""
    task_progress: Optional[float] = None
