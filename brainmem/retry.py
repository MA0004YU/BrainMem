"""Guided retry (Section III-B, "Guided-retry specification").

Retry is invoked only when the first attempt at an episode fails and is
controlled by one budget parameter R. Let tau(0) be the failed first attempt.
For r = 1, ..., R:

  1. the environment is reset to the episode's initial configuration;
  2. working memory is cleared while M_epi and M_sem are retained;
  3. the planner re-executes the task under the same step budget, receiving
     the failure reasons extracted from tau(0), ..., tau(r-1) as additional
     prompt context;
  4. the loop halts as soon as the goal predicate is satisfied.

Consolidation then receives {tau(0), ..., tau(r)}. Task success is evaluated
on tau(0) only; retry outcomes never alter the reported status.

Retry cost: an episode consumes at most R + 1 rollouts, and the expected number
of additional rollouts per episode is bounded by R * (1 - SR_0).

Failure reasons are extracted deterministically from the rollout trace, so
retries add no LLM call beyond the planner calls of the retry itself.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from brainmem.types import StepRecord

# Termination causes recorded by the runner.
TERMINATION_SUCCESS = "success"
TERMINATION_STEP_BUDGET = "step_budget"
TERMINATION_INVALID_LIMIT = "invalid_action_limit"
TERMINATION_EMPTY_PLAN = "empty_plan"
TERMINATION_ENV_DONE = "env_done"

_TERMINATION_TEXT = {
    TERMINATION_STEP_BUDGET: "the step budget of {max_steps} environment steps was exhausted before the goal was satisfied",
    TERMINATION_INVALID_LIMIT: "the limit on invalid actions was reached",
    TERMINATION_EMPTY_PLAN: "the planner returned an empty plan before the goal was satisfied",
    TERMINATION_ENV_DONE: "the episode terminated without the goal being satisfied",
}


@dataclass
class RolloutTrace:
    attempt: int                      # 0 = first attempt tau(0); r >= 1 = r-th retry
    steps: List[StepRecord] = field(default_factory=list)
    success: bool = False
    task_progress: float = 0.0
    termination: str = TERMINATION_ENV_DONE
    planner_steps: int = 0
    env_steps: int = 0
    failure_reasons: List[str] = field(default_factory=list)


def expected_retry_bound(max_retries: int, first_attempt_success_rate: float) -> float:
    """Upper bound R * (1 - SR_0) on the expected number of retries per episode."""
    return max_retries * (1.0 - first_attempt_success_rate)


def extract_failure_reasons(trace: RolloutTrace, max_steps: int, max_items: int = 5) -> List[str]:
    """Failure reasons of a failed rollout, used as retry context and in consolidation."""
    reasons: List[str] = []
    seen = set()
    for s in trace.steps:
        if s.success is False and s.action not in seen:
            seen.add(s.action)
            msg = f"'{s.action}' failed at step {s.t}"
            if s.feedback:
                msg += f" (feedback: {s.feedback})"
            reasons.append(msg)
        if len(reasons) >= max_items:
            break
    counts = Counter(s.action for s in trace.steps)
    for action, n in counts.most_common():
        if n >= 3:
            reasons.append(f"'{action}' was executed {n} times without completing the task")
    cause = _TERMINATION_TEXT.get(trace.termination)
    if cause:
        reasons.append(cause.format(max_steps=max_steps))
    return reasons


def render_retry_context(previous: List[RolloutTrace]) -> str:
    lines = ["This is a retry of a task whose previous attempt(s) failed. "
             "Avoid repeating the failures below:"]
    for tr in previous:
        lines.append(f"Attempt {tr.attempt + 1}:")
        lines.extend(f"- {r}" for r in tr.failure_reasons)
    return "\n".join(lines)


def first_successful(traces: List[RolloutTrace]) -> Optional[RolloutTrace]:
    for tr in traces:
        if tr.success:
            return tr
    return None
