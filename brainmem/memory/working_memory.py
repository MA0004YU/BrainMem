"""Working Memory M_work (Section III-B-1).

M_work keeps a sliding window of size W over the most recent interaction
steps of the current episode. Each step record is the tuple

    m_t = (Action_t, Feedback_t, State_t, Location_t, t).

When serialized into the planner prompt, the W most recent entries are
rendered as a concise textual history together with aggregate statistics
(cumulative success rate), the current robot state, and consistency warnings
for invalid action sequences. The window is cleared at the start of every
episode and at the start of every guided retry.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional


@dataclass
class WorkingMemoryEntry:
    action: str
    feedback: str            # empty when feedback is not recorded
    success: Optional[bool]  # None when feedback is not recorded
    state: str
    location: str
    t: int


# A consistency checker receives the window (oldest first) and the current
# robot state, and returns human-readable warnings.
ConsistencyChecker = Callable[[List[WorkingMemoryEntry], str], List[str]]


class WorkingMemory:
    def __init__(self, window_size: int, include_feedback: bool = True,
                 include_success_state: bool = True, include_agent_state: bool = True,
                 include_consistency_check: bool = True,
                 consistency_checker: Optional[ConsistencyChecker] = None):
        self.window_size = window_size
        self.include_feedback = include_feedback
        # The cumulative success rate is computed from execution feedback,
        # so it is only available when feedback is recorded.
        self.include_success_state = include_success_state and include_feedback
        self.include_agent_state = include_agent_state
        self.include_consistency_check = include_consistency_check
        self.consistency_checker = consistency_checker
        self._window: Deque[WorkingMemoryEntry] = deque(maxlen=window_size)
        self._num_steps = 0
        self._num_success = 0
        self._current_state = ""
        self._current_location = ""

    # -- step-level update ---------------------------------------------------
    def clear(self, initial_state: str = "", initial_location: str = "") -> None:
        self._window.clear()
        self._num_steps = 0
        self._num_success = 0
        self._current_state = initial_state
        self._current_location = initial_location

    def append(self, entry: WorkingMemoryEntry) -> None:
        """Append m_t; the deque discards the oldest entry beyond W."""
        self._window.append(entry)
        self._num_steps += 1
        self._num_success += int(entry.success is True)
        self._current_state = entry.state
        self._current_location = entry.location

    # -- retrieval -------------------------------------------------------------
    @property
    def entries(self) -> List[WorkingMemoryEntry]:
        return list(self._window)

    def __len__(self) -> int:
        return len(self._window)

    def consistency_warnings(self) -> List[str]:
        if not (self.include_consistency_check and self.consistency_checker and self._window):
            return []
        return self.consistency_checker(self.entries, self._current_state)

    def render(self) -> str:
        lines: List[str] = []
        if self._window:
            lines.append(f"Recent interaction history (last {len(self._window)} of at most {self.window_size} steps):")
            for e in self._window:
                parts = [f"Step {e.t}", f"Action: {e.action}"]
                if self.include_feedback and e.success is not None:
                    parts.append(f"Feedback: {'success' if e.success else 'failure'} ({e.feedback})")
                if self.include_agent_state:
                    parts.append(f"State: {e.state}")
                parts.append(f"Location: {e.location}")
                lines.append("- " + " | ".join(parts))
        else:
            lines.append("No action has been executed in this episode yet.")
        if self.include_success_state and self._num_steps:
            rate = self._num_success / self._num_steps
            lines.append(f"Cumulative action success rate: {self._num_success}/{self._num_steps} ({rate:.0%}).")
        if self.include_agent_state and self._current_state:
            lines.append(f"Current robot state: {self._current_state}; location: {self._current_location}.")
        warnings = self.consistency_warnings()
        if warnings:
            lines.append("Consistency warnings:")
            lines.extend(f"- {w}" for w in warnings)
        return "\n".join(lines)
