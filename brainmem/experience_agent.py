"""Experience agent (Section III-A, III-B-3, III-D step 3-4).

The experience agent is independent of the planner. After an episode
terminates it makes exactly one LLM call that

  * condenses the episode into an experience record (summarized action
    sequence, key failure reasons, key success pattern, learning insights);
  * distills reusable guidelines from the episode together with compatible
    experience records already in semantic memory.

Guidelines are only distilled when some rollout of the episode succeeded:
from tau(0) if the first attempt succeeded, otherwise from the contrast
between the terminal successful retry and its failed predecessors. If no
rollout succeeded, only the failure reasons are recorded.

The call runs after the episode terminates and therefore never delays action
selection.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from brainmem.llm import BackboneLLM, Usage
from brainmem.retry import RolloutTrace, first_successful
from brainmem.types import TaskQuery
from brainmem.utils import extract_json_object

SYSTEM_PROMPT = """You are the experience agent of an embodied robot. You analyze execution logs of household and manipulation tasks and turn them into compact, reusable knowledge for future planning.
Return a single JSON object and nothing else, with the keys:
{
  "action_sequence_summary": "one sentence summarizing what the robot did",
  "success_pattern": "the key decision or action ordering that led to success, or an empty string",
  "failure_reasons": ["root causes of failed attempts; empty list if the first attempt succeeded"],
  "learning_insights": ["general principles that transfer to other tasks of the same type"],
  "guidelines": [{"text": "an imperative rule, e.g. 'Before turning on the faucet, place the object in the sink.'",
                  "task_types": ["task types the rule applies to"],
                  "objects": ["object categories the rule applies to; empty if generic"]}]
}
Rules:
- Be concrete: refer to actions, objects and preconditions, not to generic advice.
- Each guideline must be supported by the logs. Use at most 3 guidelines.
- If no attempt succeeded, return an empty "guidelines" list and an empty "success_pattern"."""


def _render_trace(tr: RolloutTrace, with_feedback: bool) -> str:
    head = f"Attempt {tr.attempt + 1} ({'first attempt' if tr.attempt == 0 else 'retry'}): " \
           f"{'SUCCESS' if tr.success else 'FAILURE'}, {tr.env_steps} environment steps"
    lines = [head]
    for s in tr.steps:
        if with_feedback:
            lines.append(f"  {s.t}. {s.action} -> {'success' if s.success else 'failure'}: {s.feedback}")
        else:
            lines.append(f"  {s.t}. {s.action}")
    if not tr.steps:
        lines.append("  (no action executed)")
    return "\n".join(lines)


def _render_record(r: Dict[str, Any]) -> str:
    parts = [f"Task: {r['instruction']} (outcome: {'success' if r['success'] else 'failure'})"]
    if r.get("success_pattern"):
        parts.append(f"success pattern: {r['success_pattern']}")
    if r.get("failure_reasons"):
        parts.append("failure reasons: " + "; ".join(r["failure_reasons"]))
    if r.get("learning_insights"):
        parts.append("insights: " + "; ".join(r["learning_insights"]))
    return " | ".join(parts)


class ExperienceAgent:
    def __init__(self, llm: BackboneLLM, use_feedback: bool = True):
        self.llm = llm
        self.use_feedback = use_feedback

    def build_prompt(self, query: TaskQuery, traces: Sequence[RolloutTrace],
                     compatible_records: Sequence[Dict[str, Any]]) -> str:
        success = first_successful(list(traces))
        if traces[0].success:
            mode = ("The first attempt succeeded. Extract the success pattern and distill guidelines from it "
                    "and from the compatible past experiences.")
        elif success is not None:
            mode = (f"The first attempt failed and retry attempt {success.attempt + 1} succeeded. Extract the failure "
                    "reasons of the failed attempts and distill corrective guidelines from the difference between the "
                    "successful attempt and the failed attempts before it.")
        else:
            mode = "No attempt succeeded. Extract the failure reasons; do not produce guidelines."
        blocks = [
            f"Task instruction: {query.instruction}",
            f"Task type: {query.task_type}; target object: {query.target_object or 'n/a'}; room: {query.room_id}",
            mode,
            "Execution logs:",
            "\n".join(_render_trace(t, self.use_feedback) for t in traces),
        ]
        if compatible_records:
            blocks.append("Compatible past experiences of the same task type:")
            blocks.extend(f"- {_render_record(r)}" for r in compatible_records)
        return "\n\n".join(blocks)

    def consolidate(self, query: TaskQuery, traces: Sequence[RolloutTrace],
                    compatible_records: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Usage]:
        """Return the parsed analysis and the token usage of the single LLM call."""
        prompt = self.build_prompt(query, traces, compatible_records)
        text, usage = self.llm.complete(SYSTEM_PROMPT, prompt)
        try:
            parsed = extract_json_object(text)
        except (ValueError, json.JSONDecodeError):
            parsed = {}
        analysis = self._normalize(parsed, traces)
        analysis["raw_response"] = text
        return analysis, usage

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    def _normalize(self, parsed: Dict[str, Any], traces: Sequence[RolloutTrace]) -> Dict[str, Any]:
        any_success = first_successful(list(traces)) is not None
        guidelines: List[Dict[str, Any]] = []
        if any_success:
            for g in parsed.get("guidelines", []) or []:
                if isinstance(g, str):
                    g = {"text": g}
                if not isinstance(g, dict) or not str(g.get("text", "")).strip():
                    continue
                guidelines.append({"text": str(g["text"]).strip(),
                                   "task_types": self._as_list(g.get("task_types")),
                                   "objects": [o.lower() for o in self._as_list(g.get("objects"))]})
        failure_reasons = self._as_list(parsed.get("failure_reasons"))
        if not traces[0].success and not failure_reasons:
            # The rule-based reasons of the failed attempts are always available.
            failure_reasons = [r for t in traces if not t.success for r in t.failure_reasons]
        return {
            "action_sequence_summary": str(parsed.get("action_sequence_summary", "")).strip(),
            "success_pattern": str(parsed.get("success_pattern", "")).strip() if any_success else "",
            "failure_reasons": failure_reasons if not traces[0].success else [],
            "learning_insights": self._as_list(parsed.get("learning_insights")),
            "guidelines": guidelines[:3],
        }
