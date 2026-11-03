"""BrainMem: the memory system that wraps a planner (Sections III-B to III-D).

Step-level updates (after every action)
    1. Working memory: append m_t, discarding the oldest entry beyond W.
    2. Trajectory KG: create or reuse s_t and add e_{t-1->t}.
    3. Spatial KG: add newly observed objects/relations, increment support.

Episode-level updates (after each episode)
    1. Finalize the Trajectory KG with the outcome (done per rollout).
    2. Promote frequently confirmed spatial relations.
    3. Experience agent: summarize the episode into an experience record.
    4. Update/create guidelines, recompute utility scores, prune.

Hierarchical retrieval (Section III-C) for the query Q = {room, task_type, object}
    Working memory  -> the whole window
    Trajectory KG   -> action sequences of highly similar past tasks, failure-tagged transitions
    Spatial KG      -> relations of the same room
    Semantic memory -> top-k guidelines and experience records by task type and object category
The retrieved parts are serialized to text and appended to the planner prompt.
Neither retrieval nor step-level updates invoke the LLM.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from brainmem.config import BrainMemConfig
from brainmem.experience_agent import ExperienceAgent
from brainmem.memory.semantic_memory import SemanticMemory
from brainmem.memory.spatial_kg import SpatialKG
from brainmem.memory.trajectory_kg import TrajectoryKG
from brainmem.memory.working_memory import ConsistencyChecker, WorkingMemory, WorkingMemoryEntry
from brainmem.retry import RolloutTrace, render_retry_context
from brainmem.types import StepObservation, StepRecord, TaskQuery


class BrainMem:
    def __init__(self, cfg: BrainMemConfig, consistency_checker: Optional[ConsistencyChecker] = None,
                 experience_agent: Optional[ExperienceAgent] = None):
        cfg.validate()
        self.cfg = cfg
        os.makedirs(cfg.memory_dir, exist_ok=True)
        self.wm = WorkingMemory(cfg.window_size, include_feedback=cfg.use_feedback,
                                include_success_state=cfg.wm_success_state,
                                include_agent_state=cfg.wm_agent_state,
                                include_consistency_check=cfg.wm_consistency_check,
                                consistency_checker=consistency_checker) if cfg.use_working_memory else None
        use_epi = cfg.use_episodic_memory
        self.traj = TrajectoryKG(cfg.memory_dir, cfg.state_merge_threshold, cfg.task_similarity_threshold,
                                 cfg.max_reasoning_examples) if use_epi and cfg.use_trajectory_kg else None
        self.spatial = SpatialKG(cfg.memory_dir, cfg.spatial_stable_support) if use_epi and cfg.use_spatial_kg else None
        self.sem = SemanticMemory(cfg.memory_dir, cfg.alpha, cfg.beta, cfg.usage_saturation, cfg.max_guidelines,
                                  cfg.prune_trigger, cfg.protect_confidence, cfg.protect_min_success,
                                  cfg.guideline_merge_threshold) if cfg.use_semantic_memory else None
        self.agent = experience_agent if cfg.use_semantic_memory else None
        if self.sem is not None and self.agent is None:
            raise ValueError("semantic memory requires an ExperienceAgent")
        self._retry_context = ""
        self._applied_guidelines: List[str] = []
        self._attempt = 0

    # ================================================================== episode / rollout
    def begin_episode(self, query: TaskQuery) -> None:
        self._applied_guidelines = []
        if self.spatial is not None:
            self.spatial.begin_episode(query.episode_key)

    def begin_rollout(self, query: TaskQuery, obs0: StepObservation, attempt: int,
                      previous: Sequence[RolloutTrace] = ()) -> None:
        """Start tau(attempt). Working memory is cleared; M_epi and M_sem are retained."""
        self._attempt = attempt
        self._retry_context = render_retry_context(list(previous)) if attempt > 0 and previous else ""
        if self.wm is not None:
            self.wm.clear(obs0.agent_state, obs0.location)
        if self.traj is not None:
            self.traj.start_rollout(query.episode_key, attempt, query.task_type, query.target_object,
                                    query.instruction, obs0.room_id, obs0.summary, obs0.state_tokens)
        if self.spatial is not None:
            self.spatial.observe(obs0.room_id, obs0.scene_id, obs0.objects, obs0.relations)

    def after_step(self, step: StepRecord, obs: StepObservation) -> None:
        """Step-level updates."""
        success = step.success if self.cfg.use_feedback else None
        feedback = step.feedback if self.cfg.use_feedback else ""
        if self.wm is not None:
            self.wm.append(WorkingMemoryEntry(step.action, feedback, success, obs.agent_state, obs.location, step.t))
        if self.traj is not None:
            self.traj.add_step(step.action, success, obs.room_id, obs.summary, obs.state_tokens,
                               reasoning=step.reasoning, feedback=feedback)
        if self.spatial is not None:
            self.spatial.observe(obs.room_id, obs.scene_id, obs.objects, obs.relations)

    def end_rollout(self, success: bool) -> None:
        if self.traj is not None:
            self.traj.end_rollout(success if self.cfg.use_feedback else None)

    # ================================================================== retrieval
    def context(self, query: TaskQuery, room_id: str, scene_id: str) -> str:
        """Hierarchically retrieved memory context for one planning call."""
        cfg = self.cfg
        sections: List[str] = []

        if self.wm is not None:
            sections.append("### Working memory\n" + self.wm.render())

        if self.traj is not None:
            lines: List[str] = []
            seqs = self.traj.recall_sequences(query.task_type, query.target_object, query.instruction,
                                              cfg.traj_recall_sequences, use_outcome=cfg.use_feedback)
            if seqs:
                title = ("Successful action sequences of similar past tasks (ranked by success count):"
                         if cfg.use_feedback else "Action sequences of similar past tasks:")
                lines.append(title)
                for i, (seq, count) in enumerate(seqs, 1):
                    shown = seq[:cfg.traj_recall_max_actions]
                    more = " -> ..." if len(seq) > len(shown) else ""
                    lines.append(f"{i}. [x{count}] " + " -> ".join(shown) + more)
            if cfg.use_feedback:
                failed = self.traj.failed_transitions(query.task_type, room_id, self.traj.current_state,
                                                      cfg.failed_transitions_shown)
                if failed:
                    lines.append("Known failed actions for this task type (avoid them):")
                    for f in failed:
                        fb = f" -- {f['feedback']}" if f["feedback"] else ""
                        lines.append(f"- '{f['action']}' failed {f['failures']}x{fb}")
            if lines:
                sections.append("### Episodic memory: trajectory knowledge graph\n" + "\n".join(lines))

        if self.spatial is not None:
            sp = self.spatial.retrieve(room_id, scene_id, cfg.spatial_relations_shown)
            lines = []
            if sp["contains"]:
                objs = ", ".join(r["object"] for r in sp["contains"])
                lines.append(f"{room_id} contains: {objs}")
            for r in sp["relations"]:
                n = r["support"]
                lines.append(f"- {r['subject']} {r['predicate']} {r['object']} "
                             f"(confirmed in {n} episode{'s' if n != 1 else ''})")
            if lines:
                sections.append("### Episodic memory: spatial knowledge graph\n" + "\n".join(lines))

        if self.sem is not None:
            lines = []
            if cfg.sm_learning_insights:
                guidelines = self.sem.retrieve_guidelines(query.task_type, query.target_object, cfg.top_k)
                if guidelines:
                    lines.append("Guidelines (ranked by utility):")
                    for i, g in enumerate(guidelines, 1):
                        lines.append(f"{i}. {g['text']} [utility {g['utility']:.2f}, "
                                     f"succeeded {g['n_success']}/{g['n_total']}]")
                    if self._attempt == 0:
                        for g in guidelines:
                            if g["guideline_id"] not in self._applied_guidelines:
                                self._applied_guidelines.append(g["guideline_id"])
            records = self.sem.retrieve_records(query.task_type, query.target_object, query.room_id, cfg.top_k,
                                                exclude_episode=query.episode_key)
            rec_lines = []
            for r in records:
                parts = [f"Task '{r['instruction']}' ({'success' if r['success'] else 'failure'})"]
                if cfg.sm_success_patterns and r.get("success_pattern"):
                    parts.append(f"success pattern: {r['success_pattern']}")
                if cfg.sm_failure_reasons and r.get("failure_reasons"):
                    parts.append("failure reasons: " + "; ".join(r["failure_reasons"][:3]))
                if cfg.sm_learning_insights and r.get("learning_insights"):
                    parts.append("insights: " + "; ".join(r["learning_insights"][:2]))
                if len(parts) > 1:
                    rec_lines.append("- " + " | ".join(parts))
            if rec_lines:
                lines.append("Relevant past experiences:")
                lines.extend(rec_lines)
            if lines:
                sections.append("### Semantic memory\n" + "\n".join(lines))

        if self._retry_context:
            sections.append("### Retry context\n" + self._retry_context)

        if not sections:
            return ""
        return ("## BrainMem memory context\n"
                "Use the memory below together with the current observation when choosing the next actions.\n\n"
                + "\n\n".join(sections))

    # ================================================================== consolidation
    def consolidate(self, query: TaskQuery, traces: Sequence[RolloutTrace]) -> Dict[str, Any]:
        """Episode-level updates 2-4. Returns a log of what changed (and the LLM usage)."""
        out: Dict[str, Any] = {"llm_usage": {"input_tokens": 0, "output_tokens": 0},
                               "guidelines_added": [], "guidelines_pruned": [], "record_id": None}
        first_success = bool(traces[0].success)
        if self.spatial is not None:
            self.spatial.consolidate()
        if self.sem is not None:
            compatible = self.sem.retrieve_records(query.task_type, query.target_object, query.room_id,
                                                   self.cfg.distill_context_records,
                                                   exclude_episode=query.episode_key)
            analysis, usage = self.agent.consolidate(query, traces, compatible)
            out["llm_usage"] = usage
            recovered = next((t.attempt for t in traces if t.success), None)
            record = self.sem.add_record({
                "episode_key": query.episode_key, "instruction": query.instruction, "room_id": query.room_id,
                "task_type": query.task_type, "target_object": query.target_object,
                "success": first_success, "recovered_on_attempt": None if first_success else recovered,
                "attempts": len(traces),
                "action_sequence": [s.action for s in (traces[recovered].steps if recovered is not None else traces[0].steps)],
                "action_sequence_summary": analysis["action_sequence_summary"],
                "success_pattern": analysis["success_pattern"],
                "failure_reasons": analysis["failure_reasons"],
                "learning_insights": analysis["learning_insights"],
            })
            out["record_id"] = record["record_id"]
            if self.cfg.use_feedback:
                self.sem.record_application(self._applied_guidelines, first_success)
            for g in analysis["guidelines"]:
                task_types = g["task_types"] or [query.task_type]
                if query.task_type not in task_types:
                    task_types = task_types + [query.task_type]
                objects = g["objects"] or ([query.target_object] if query.target_object else [])
                added = self.sem.add_or_merge_guideline(g["text"], task_types, objects, record["record_id"])
                out["guidelines_added"].append(added["guideline_id"])
            out["guidelines_pruned"] = self.sem.prune()
            out["analysis"] = analysis
        out["applied_guidelines"] = list(self._applied_guidelines)
        return out

    # ================================================================== persistence / stats
    def save(self) -> None:
        for store in (self.traj, self.spatial, self.sem):
            if store is not None:
                store.save()

    def stats(self) -> Dict[str, Any]:
        s: Dict[str, Any] = {}
        for store in (self.traj, self.spatial, self.sem):
            if store is not None:
                s.update(store.stats())
        total = 0
        for name in os.listdir(self.cfg.memory_dir):
            p = os.path.join(self.cfg.memory_dir, name)
            if os.path.isfile(p):
                total += os.path.getsize(p)
        s["storage_bytes"] = total
        return s
