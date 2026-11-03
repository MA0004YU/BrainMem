"""Memory-augmented EmbodiedBench planners.

BrainMem is plug-and-play: the EmbodiedBench planner classes (VLMPlanner,
EBNavigationPlanner, ManipPlanner) are used unchanged, and the retrieved
memory context is appended to the prompt that their ``process_prompt`` builds.
``augment_planner`` re-classes an existing planner instance into a subclass
that overrides only ``process_prompt``.
"""
from __future__ import annotations

from typing import Any, Dict, Type

_CACHE: Dict[type, type] = {}


def _memory_subclass(base: Type) -> Type:
    if base in _CACHE:
        return _CACHE[base]

    class MemoryAugmented(base):  # type: ignore[misc, valid-type]
        memory_context: str = ""

        def process_prompt(self, *args, **kwargs):
            prompt = super().process_prompt(*args, **kwargs)
            if not self.memory_context:
                return prompt
            if isinstance(prompt, tuple):  # ManipPlanner returns (general_prompt, task_prompt)
                general, task = prompt
                return general + "\n\n" + self.memory_context, task
            return prompt + "\n\n" + self.memory_context

    MemoryAugmented.__name__ = f"BrainMem{base.__name__}"
    MemoryAugmented.__qualname__ = MemoryAugmented.__name__
    _CACHE[base] = MemoryAugmented
    return MemoryAugmented


def augment_planner(planner: Any) -> Any:
    if not getattr(planner, "_brainmem_augmented", False):
        planner.__class__ = _memory_subclass(type(planner))
        planner._brainmem_augmented = True
        planner.memory_context = ""
    return planner


def set_memory_context(planner: Any, context: str) -> None:
    planner.memory_context = context
