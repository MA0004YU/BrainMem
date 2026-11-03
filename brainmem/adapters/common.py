"""Helpers shared by the benchmark adapters."""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

from brainmem.memory.working_memory import WorkingMemoryEntry


def room_type_from_floorplan(scene: str) -> str:
    """AI2-THOR scene naming: FloorPlan1-30 kitchens, 201-230 living rooms,
    301-330 bedrooms, 401-430 bathrooms."""
    m = re.match(r"FloorPlan(\d+)", scene or "")
    if not m:
        return "room"
    n = int(m.group(1))
    if n < 100:
        return "kitchen"
    return {2: "living room", 3: "bedroom", 4: "bathroom"}.get(n // 100, "room")


def count_recent(window: List[WorkingMemoryEntry], prefix: str, min_count: int) -> List[Tuple[str, int]]:
    """Actions starting with ``prefix`` executed at least ``min_count`` times in the window."""
    c = Counter(e.action for e in window if e.action.lower().startswith(prefix))
    return [(a, n) for a, n in c.most_common() if n >= min_count]


def repeated_failures(window: List[WorkingMemoryEntry]) -> List[str]:
    """Actions that failed at least twice in the window."""
    c = Counter(e.action for e in window if e.success is False)
    return [f"'{a}' failed {n} times in the last {len(window)} steps; choose a different action."
            for a, n in c.most_common() if n >= 2]


def repeated_identical(window: List[WorkingMemoryEntry], min_run: int = 3) -> List[str]:
    """The same action issued ``min_run`` or more times in a row at the end of the window."""
    if len(window) < min_run:
        return []
    last = window[-1].action
    run = 0
    for e in reversed(window):
        if e.action != last:
            break
        run += 1
    if run >= min_run:
        return [f"'{last}' was issued {run} times in a row; the agent is not making progress."]
    return []
