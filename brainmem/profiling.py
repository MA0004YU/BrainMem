"""Wall-clock and token accounting used for Section IV-D and Table XV.

Timing categories (seconds, wall clock):
    planner_inference  planner.act(), i.e. the LLM planning call incl. network time
    env_simulation     env.reset() and env.step()
    retrieval          hierarchical memory retrieval and prompt serialization
    step_update        step-level memory updates (incl. state summarization)
    consolidation      episode-level consolidation (experience agent call, pruning, saving)

Tokens are read from the provider responses (``usage`` fields) by wrapping the
SDK client held by the EmbodiedBench planner, and are attributed to the phase
that is active when the call is made: ``attempt_0`` (tau(0)), ``retry``
(tau(r), r >= 1) or ``consolidation``.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

CATEGORIES = ("planner_inference", "env_simulation", "retrieval", "step_update", "consolidation")
PHASES = ("attempt_0", "retry", "consolidation")


class Profiler:
    def __init__(self) -> None:
        self.reset_episode()

    def reset_episode(self) -> None:
        self.phase = "attempt_0"
        self.seconds: Dict[str, Dict[str, float]] = {p: defaultdict(float) for p in PHASES}
        self.tokens: Dict[str, Dict[str, int]] = {p: {"input_tokens": 0, "output_tokens": 0, "calls": 0}
                                                  for p in PHASES}
        self.decision_latencies: Dict[str, List[float]] = {p: [] for p in PHASES}
        self.retrieval_latencies: List[float] = []
        self.update_latencies: List[float] = []

    @contextmanager
    def section(self, category: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.seconds[self.phase][category] += dt
            if category == "retrieval":
                self.retrieval_latencies.append(dt)
            elif category == "step_update":
                self.update_latencies.append(dt)

    def add_decision(self, seconds: float) -> None:
        self.decision_latencies[self.phase].append(seconds)

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        t = self.tokens[self.phase]
        t["input_tokens"] += int(input_tokens or 0)
        t["output_tokens"] += int(output_tokens or 0)
        t["calls"] += 1

    def episode_summary(self) -> Dict[str, Any]:
        return {
            "seconds": {p: dict(v) for p, v in self.seconds.items()},
            "tokens": self.tokens,
            "decision_latency_s": self.decision_latencies,
            "retrieval_latency_s": self.retrieval_latencies,
            "update_latency_s": self.update_latencies,
        }


# ---------------------------------------------------------------------- token metering
_MARK = "_brainmem_metered"


def _wrap_method(owner: Any, name: str, profiler: Profiler, kind: str) -> None:
    fn = getattr(owner, name, None)
    if fn is None or getattr(fn, _MARK, False):
        return

    def wrapped(*args, **kwargs):
        response = fn(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            if kind == "anthropic":
                profiler.add_tokens(getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
            else:
                profiler.add_tokens(getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))
        return response

    setattr(wrapped, _MARK, True)
    setattr(owner, name, wrapped)


class _MeteredPipeline:
    """Proxy for an LMDeploy pipeline that records per-call token counts."""

    def __init__(self, pipe: Any, profiler: Profiler):
        self._pipe = pipe
        self._profiler = profiler
        setattr(self, _MARK, True)

    def __call__(self, *args, **kwargs):
        response = self._pipe(*args, **kwargs)
        items = response if isinstance(response, list) else [response]
        for r in items:
            self._profiler.add_tokens(getattr(r, "input_token_len", 0), getattr(r, "generate_token_len", 0))
        return response

    def __getattr__(self, item):
        return getattr(self._pipe, item)


def attach_token_meter(holder: Any, attr: str, profiler: Profiler) -> None:
    """Meter the SDK client stored at ``holder.<attr>`` (e.g. RemoteModel.model)."""
    client = getattr(holder, attr, None)
    if client is None or getattr(client, _MARK, False):
        return
    if type(client).__module__.startswith("lmdeploy"):
        setattr(holder, attr, _MeteredPipeline(client, profiler))
        return
    messages = getattr(client, "messages", None)
    if messages is not None and hasattr(messages, "create"):
        _wrap_method(messages, "create", profiler, "anthropic")
    chat = getattr(client, "chat", None)
    if chat is not None:
        _wrap_method(chat.completions, "create", profiler, "openai")
    beta = getattr(client, "beta", None)
    if beta is not None and getattr(beta, "chat", None) is not None:
        _wrap_method(beta.chat.completions, "parse", profiler, "openai")
    try:
        setattr(client, _MARK, True)
    except AttributeError:
        pass
