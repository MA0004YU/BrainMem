"""Episode loop shared by all benchmarks.

``BrainMemEvaluatorMixin`` is combined with an EmbodiedBench evaluator class
(e.g. ``EB_AlfredEvaluator``). EmbodiedBench's own ``evaluate_main`` builds the
environment and planner for every eval set exactly as in the original
benchmark; the mixin replaces only ``evaluate`` (the per-episode loop) with the
BrainMem loop:

    for each episode:
        tau(0) <- rollout(attempt 0)                       # reported result
        if tau(0) failed and feedback is enabled:           # guided retry
            for r = 1..R: tau(r) <- rollout(attempt r, failure reasons of tau(0..r-1))
                          stop when tau(r) succeeds
        consolidate({tau(0), ..., tau(r)})                 # episode-level updates

With the ``base`` preset BrainMem is disabled and the loop performs a single
rollout without memory, i.e. the original EmbodiedBench protocol.

Benchmark subclasses implement the hooks in the "benchmark-specific" block.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brainmem.brainmem import BrainMem
from brainmem.config import BrainMemConfig
from brainmem.experience_agent import ExperienceAgent
from brainmem.llm import BackboneLLM
from brainmem.memory.working_memory import ConsistencyChecker
from brainmem.planner import augment_planner, set_memory_context
from brainmem.profiling import Profiler, attach_token_meter
from brainmem.retry import (TERMINATION_EMPTY_PLAN, TERMINATION_ENV_DONE, TERMINATION_INVALID_LIMIT,
                            TERMINATION_STEP_BUDGET, TERMINATION_SUCCESS, RolloutTrace,
                            extract_failure_reasons)
from brainmem.types import StepObservation, StepRecord, TaskQuery

MAX_CONSECUTIVE_PLANNER_ERRORS = 5


class BrainMemEvaluatorMixin:
    benchmark: str = ""

    # ------------------------------------------------------------------ setup
    def setup_brainmem(self, bm_cfg: BrainMemConfig, run_dir: str, fresh_memory: bool = True) -> None:
        self.bm_cfg = bm_cfg
        self.bm_run_dir = run_dir
        if not bm_cfg.memory_dir:
            bm_cfg.memory_dir = os.path.join(run_dir, "memory")
        if fresh_memory and os.path.isdir(bm_cfg.memory_dir):
            shutil.rmtree(bm_cfg.memory_dir)
        os.makedirs(bm_cfg.memory_dir, exist_ok=True)
        self.bm: Optional[BrainMem] = None
        self.profiler = Profiler()
        with open(os.path.join(run_dir, "brainmem_config.json"), "w") as f:
            json.dump(bm_cfg.to_dict(), f, indent=2)

    def _ensure_brainmem(self) -> None:
        """Memory is created once per run and persists across the eval sets of a benchmark."""
        attach_token_meter(self.planner.model, "model", self.profiler)
        if not self.bm_cfg.enabled:
            return
        augment_planner(self.planner)
        if self.bm is None:
            agent = None
            if self.bm_cfg.use_semantic_memory:
                llm = BackboneLLM.from_planner(self.planner, self.bm_cfg.experience_model,
                                               self.bm_cfg.experience_max_tokens, self.bm_cfg.experience_temperature)
                attach_token_meter(llm, "client", self.profiler)
                agent = ExperienceAgent(llm, use_feedback=self.bm_cfg.use_feedback)
            self.bm = BrainMem(self.bm_cfg, self.consistency_checker(), agent)

    # ------------------------------------------------------------------ benchmark-specific hooks
    def consistency_checker(self) -> Optional[ConsistencyChecker]:
        raise NotImplementedError

    def episode_index(self) -> int:
        """Index used in result file names (after env.reset())."""
        raise NotImplementedError

    def rewind_episode(self) -> None:
        """Make the next env.reset() restore the current episode's initial configuration."""
        raise NotImplementedError

    def task_query(self) -> TaskQuery:
        raise NotImplementedError

    def observe(self, obs: Any, info: Optional[Dict[str, Any]], action: Optional[str] = None,
                success: Optional[bool] = None) -> StepObservation:
        raise NotImplementedError

    def run_rollout(self, attempt: int, query_holder: Dict[str, Any], previous: List[RolloutTrace]) -> RolloutTrace:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers for run_rollout
    def plan(self, query: TaskQuery, act_fn, *args) -> Tuple[Any, str]:
        """Retrieve memory, set it into the planner prompt, call the planner."""
        t0 = time.perf_counter()
        if self.bm is not None:
            with self.profiler.section("retrieval"):
                obs_room, obs_scene = self._current_room
                set_memory_context(self.planner, self.bm.context(query, obs_room, obs_scene))
        with self.profiler.section("planner_inference"):
            out = act_fn(*args)
        self.profiler.add_decision(time.perf_counter() - t0)
        return out

    def env_reset(self, *args, **kwargs):
        with self.profiler.section("env_simulation"):
            return self.env.reset(*args, **kwargs)

    def env_step(self, *args, **kwargs):
        with self.profiler.section("env_simulation"):
            return self.env.step(*args, **kwargs)

    def begin_rollout(self, query: TaskQuery, obs: Any, attempt: int, previous: List[RolloutTrace]) -> None:
        """Clear working memory and create s_0 from the reset observation."""
        if self.bm is None:
            return
        with self.profiler.section("step_update"):
            obs0 = self.observe(obs, None)
            self._current_room = (obs0.room_id, obs0.scene_id)
            self.bm.begin_rollout(query, obs0, attempt, previous)

    def record_step(self, trace: RolloutTrace, obs: Any, info: Dict[str, Any], action_str: str,
                    reasoning: str) -> None:
        success = bool(info.get("last_action_success", info.get("action_success", 0)))
        step = StepRecord(t=int(info.get("env_step", len(trace.steps) + 1)), action=action_str,
                          feedback=str(info.get("env_feedback", "")), success=success,
                          reasoning=_reasoning_text(reasoning),
                          task_progress=_as_float(info.get("task_progress")))
        trace.steps.append(step)
        if self.bm is not None:
            with self.profiler.section("step_update"):
                st = self.observe(obs, info, action_str, success)
                self._current_room = (st.room_id, st.scene_id)
                self.bm.after_step(step, st)

    def finish_trace(self, trace: RolloutTrace, info: Dict[str, Any], empty_plan: bool) -> RolloutTrace:
        trace.success = bool(_as_float(info.get("task_success")) or 0.0)
        trace.task_progress = _as_float(info.get("task_progress")) or (1.0 if trace.success else 0.0)
        trace.planner_steps = int(getattr(self.planner, "planner_steps", 0))
        trace.env_steps = int(self.env._current_step)
        if trace.success:
            trace.termination = TERMINATION_SUCCESS
        elif empty_plan:
            trace.termination = TERMINATION_EMPTY_PLAN
        elif getattr(self.env, "_cur_invalid_actions", 0) >= getattr(self.env, "_max_invalid_actions", 1 << 30):
            trace.termination = TERMINATION_INVALID_LIMIT
        elif self.env._current_step >= self.env._max_episode_steps:
            trace.termination = TERMINATION_STEP_BUDGET
        else:
            trace.termination = TERMINATION_ENV_DONE
        if not trace.success:
            trace.failure_reasons = extract_failure_reasons(trace, self.env._max_episode_steps)
        if self.bm is not None:
            with self.profiler.section("step_update"):
                self.bm.end_rollout(trace.success)
        return trace

    # ------------------------------------------------------------------ main loop
    def evaluate(self):
        from tqdm import tqdm
        self._ensure_brainmem()
        self.prepare_eval_set()
        base_log_path = self.env.log_path
        progress = tqdm(total=self.env.number_of_episodes, desc=f"BrainMem {self.benchmark} {self.eval_set}")
        while self.env._current_episode_num < self.env.number_of_episodes:
            self.profiler.reset_episode()
            wall0 = time.perf_counter()
            traces: List[RolloutTrace] = []
            holder: Dict[str, Any] = {}
            first = self.run_rollout(0, holder, traces)
            traces.append(first)
            first_info = dict(holder.get("episode_info", {}))
            episode_idx = self.episode_index()
            query: TaskQuery = holder["query"]
            if self.bm is not None and self.bm_cfg.retry_enabled and not first.success:
                self.profiler.phase = "retry"
                for r in range(1, self.bm_cfg.max_retries + 1):
                    self.rewind_episode()
                    self.env.log_path = os.path.join(base_log_path, f"retry_{r}")
                    tr = self.run_rollout(r, holder, traces)
                    traces.append(tr)
                    if tr.success:
                        break
                self.env.log_path = base_log_path
            consolidation: Dict[str, Any] = {}
            if self.bm is not None:
                self.profiler.phase = "consolidation"
                with self.profiler.section("consolidation"):
                    consolidation = self.bm.consolidate(query, traces)
                    self.bm.save()
            wall = time.perf_counter() - wall0
            self.write_episode(episode_idx, query, traces, first_info, consolidation, wall)
            progress.update()
        progress.close()
        self.finish_eval_set()

    def prepare_eval_set(self) -> None:
        """Called after EmbodiedBench built env/planner for an eval set."""

    def finish_eval_set(self) -> None:
        """Called after the last episode of an eval set."""

    # ------------------------------------------------------------------ result files
    def write_episode(self, episode_idx: int, query: TaskQuery, traces: List[RolloutTrace],
                      first_info: Dict[str, Any], consolidation: Dict[str, Any], wall: float) -> None:
        prof = self.profiler.episode_summary()
        first = traces[0]
        retries = traces[1:]
        sec = prof["seconds"]
        tok = prof["tokens"]
        flat: Dict[str, Any] = {
            "instruction": query.instruction,
            # tau(0): the reported result (EmbodiedBench-compatible keys)
            "task_success": float(first.success),
            "task_progress": float(first.task_progress),
            "num_steps": first.env_steps,
            "planner_steps": first.planner_steps,
            "planner_output_error": int(first_info.get("planner_output_error", 0)),
            "num_invalid_actions": int(first_info.get("num_invalid_actions", 0)),
            "episode_elapsed_seconds": float(first_info.get("episode_elapsed_seconds", 0.0)),
            # guided retry
            "num_retries": len(retries),
            "final_success": float(any(t.success for t in traces)),
            "retry_env_steps": int(sum(t.env_steps for t in retries)),
            "retry_planner_steps": int(sum(t.planner_steps for t in retries)),
            # tokens (provider usage)
            "tokens_attempt0": tok["attempt_0"]["input_tokens"] + tok["attempt_0"]["output_tokens"],
            "tokens_retry": tok["retry"]["input_tokens"] + tok["retry"]["output_tokens"],
            "tokens_consolidation": tok["consolidation"]["input_tokens"] + tok["consolidation"]["output_tokens"],
            # wall clock (seconds)
            "wall_seconds": wall,
            "planner_seconds": sum(sec[p].get("planner_inference", 0.0) for p in sec),
            "env_seconds": sum(sec[p].get("env_simulation", 0.0) for p in sec),
            "retrieval_seconds": sum(sec[p].get("retrieval", 0.0) for p in sec),
            "update_seconds": sum(sec[p].get("step_update", 0.0) for p in sec),
            "consolidation_seconds": sec["consolidation"].get("consolidation", 0.0),
            "retry_seconds": sum(sec["retry"].values()),
            "decision_latency_attempt0_mean": _mean(prof["decision_latency_s"]["attempt_0"]),
            "retrieval_latency_mean": _mean(prof["retrieval_latency_s"]),
            "update_latency_mean": _mean(prof["update_latency_s"]),
        }
        res_dir = os.path.join(self.env.log_path, "results")
        os.makedirs(res_dir, exist_ok=True)
        with open(os.path.join(res_dir, f"episode_{episode_idx}_final_res.json"), "w", encoding="utf-8") as f:
            json.dump(flat, f, ensure_ascii=False, indent=1)

        detail = {
            "episode_idx": episode_idx, "eval_set": self.eval_set, "query": query.__dict__,
            "preset_enabled": self.bm_cfg.enabled,
            "max_retries": self.bm_cfg.max_retries if self.bm_cfg.retry_enabled else 0,
            "rollouts": [{"attempt": t.attempt, "success": t.success, "task_progress": t.task_progress,
                          "termination": t.termination, "planner_steps": t.planner_steps,
                          "env_steps": t.env_steps, "failure_reasons": t.failure_reasons,
                          "steps": [s.__dict__ for s in t.steps]} for t in traces],
            "consolidation": consolidation,
            "profile": prof,
            "memory_stats": self.bm.stats() if self.bm is not None else {},
            "timestamp": time.time(),
        }
        bm_dir = os.path.join(self.env.log_path, "brainmem")
        os.makedirs(bm_dir, exist_ok=True)
        with open(os.path.join(bm_dir, f"episode_{episode_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=1, default=str)


# ---------------------------------------------------------------------- utilities
def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        x = x[0] if x else None
        if x is None:
            return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _reasoning_text(reasoning: Any) -> str:
    if isinstance(reasoning, str):
        try:
            obj = json.loads(reasoning)
        except (ValueError, TypeError):
            return reasoning[:500]
    else:
        obj = reasoning
    if isinstance(obj, dict):
        for key in ("reasoning_and_reflection", "language_plan"):
            if obj.get(key):
                return str(obj[key])[:500]
    return str(reasoning)[:500]
