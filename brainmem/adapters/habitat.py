"""EB-Habitat adapter (Habitat-sim / Habitat-lab 0.3.0, high-level skills, 30-step budget)."""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List

from embodiedbench.evaluator.eb_habitat_evaluator import EB_HabitatEvaluator

from brainmem.adapters.base import MAX_CONSECUTIVE_PLANNER_ERRORS, BrainMemEvaluatorMixin
from brainmem.adapters.common import count_recent, repeated_failures
from brainmem.memory.working_memory import WorkingMemoryEntry
from brainmem.retry import RolloutTrace
from brainmem.types import StepObservation, TaskQuery

_OPENABLE = ("refrigerator", "fridge", "cabinet", "drawer")


def habitat_consistency(window: List[WorkingMemoryEntry], state: str) -> List[str]:
    warnings: List[str] = []
    for e in window:
        a = e.action.lower()
        if e.success is False and a.startswith("pick up") and e.state != "holding nothing":
            warnings.append(f"Step {e.t} tried to pick up an object while {e.state}; place it first.")
        if e.success is False and a.startswith("place at") and e.state == "holding nothing":
            warnings.append(f"Step {e.t} tried to place an object while holding nothing.")
    for action, n in count_recent(window, prefix="navigate to", min_count=3):
        warnings.append(f"'{action}' was executed {n} times in the last {len(window)} steps; "
                        "that location has already been explored.")
    navs = [e.action for e in window if e.action.lower().startswith("navigate to")]
    if len(navs) >= 4 and navs[-1] == navs[-3] and navs[-2] == navs[-4] and navs[-1] != navs[-2]:
        warnings.append(f"The agent is alternating between '{navs[-2]}' and '{navs[-1]}'; "
                        "explore a location that has not been visited yet.")
    warnings.extend(repeated_failures(window))
    return warnings


class BrainMemHabitatEvaluator(BrainMemEvaluatorMixin, EB_HabitatEvaluator):
    benchmark = "eb-hab"

    def __init__(self, config):
        super().__init__(config)
        self._location = ""
        self._held = ""

    def consistency_checker(self):
        return habitat_consistency

    def episode_index(self) -> int:
        return self.env._current_episode_num

    def rewind_episode(self) -> None:
        core = self.env.env.env._env  # habitat.core.env.Env
        core.current_episode = core.current_episode  # keeps the episode on the next reset()
        self.env._current_episode_num -= 1

    # ------------------------------------------------------------------ task / observation
    def _scene(self) -> str:
        scene = getattr(self.env.episode_data, "scene_id", "") or ""
        return os.path.basename(scene).split(".")[0] or "scene"

    def _pickables(self) -> List[str]:
        return [s[len("pick up the "):] for s in self.env.language_skill_set if s.startswith("pick up the ")]

    def task_query(self) -> TaskQuery:
        instr = self.env.episode_language_instruction
        low = instr.lower()
        named = [o for o in self._pickables() if re.search(r"\b" + re.escape(o.lower()) + r"\b", low)]
        n = len(dict.fromkeys(named))
        task_type = f"rearrange_{min(n, 3) if n else 'unnamed'}obj"
        if any(w in low for w in _OPENABLE):
            task_type += "_receptacle"
        scene = self._scene()
        ep_id = getattr(self.env.episode_data, "episode_id", self.env._current_episode_num)
        return TaskQuery(instruction=instr, task_type=task_type, target_object=named[0].lower() if named else "",
                         room_id=scene, scene_id=scene, episode_key=f"{self.eval_set}#{ep_id}")

    def observe(self, obs, info=None, action=None, success=None) -> StepObservation:
        scene = self._scene()
        relations, objects = [], []
        if action and success:
            a = action.lower()
            if a.startswith("navigate to the "):
                self._location = action[len("navigate to the "):]
                objects.append(self._location)
            elif a.startswith("pick up the "):
                self._held = action[len("pick up the "):]
                if self._location:
                    relations.append((self._held, "on", self._location))
            elif a.startswith("place at the "):
                target = action[len("place at the "):]
                if self._held:
                    relations.append((self._held, "on", target))
                self._held = ""
        state = f"holding {self._held}" if self._held else "holding nothing"
        location = self._location or "start position"
        summary = f"at {location}; {state}"
        tokens = [f"at:{location.lower()}", f"hold:{(self._held or 'nothing').lower()}"]
        return StepObservation(room_id=scene, scene_id=scene, summary=summary, state_tokens=tokens,
                               agent_state=state, location=location, objects=objects, relations=relations)

    # ------------------------------------------------------------------ rollout (mirrors EB_HabitatEvaluator.evaluate)
    def run_rollout(self, attempt: int, holder: Dict[str, Any], previous: List[RolloutTrace]) -> RolloutTrace:
        env, planner = self.env, self.planner
        episode_info = {"reward": [], "num_invalid_actions": 0, "empty_plan": 0}
        self._location, self._held = "", ""
        obs = self.env_reset()
        img_path = env.save_image(obs)
        user_instruction = env.episode_language_instruction
        query = self.task_query()
        if attempt == 0:
            holder["query"] = query
            if self.bm is not None:
                self.bm.begin_episode(query)
        self.begin_rollout(query, obs, attempt, previous)
        planner.reset()
        trace = RolloutTrace(attempt=attempt)
        info: Dict[str, Any] = {"task_success": 0, "task_progress": 0, "env_step": 0}
        done, empty_plan, errors = False, False, 0
        while not done:
            try:
                action, reasoning = self.plan(query, planner.act, img_path, user_instruction)
                errors = 0
            except Exception as e:
                errors += 1
                print(f"planner error ({errors}/{MAX_CONSECUTIVE_PLANNER_ERRORS}): {e}")
                if errors >= MAX_CONSECUTIVE_PLANNER_ERRORS:
                    raise
                time.sleep(30)
                continue
            if action == -2:
                episode_info["empty_plan"] = 1
                empty_plan = True
                env.episode_log.append({"last_action_success": 0.0, "action_id": -2,
                                        "action_description": "empty plan", "reasoning": reasoning})
                break
            if action == -1:
                env._cur_invalid_actions += 1
                episode_info["reward"].append(-1)
                episode_info["num_invalid_actions"] += 1
                env.episode_log.append({"last_action_success": 0.0, "action_id": -1,
                                        "action_description": "invalid action", "reasoning": reasoning})
                if env._cur_invalid_actions >= env._max_invalid_actions:
                    break
                continue
            actions = action if isinstance(action, list) else [action]
            for a in actions[:min(env._max_episode_steps - env._current_step, len(actions))]:
                obs, reward, done, info = self.env_step(a, reasoning=reasoning)
                planner.update_info(info)
                img_path = env.save_image(obs)
                episode_info["reward"].append(reward)
                episode_info["num_invalid_actions"] += (info["last_action_success"] == 0)
                self.record_step(trace, obs, info, env.language_skill_set[a], reasoning)
                if done or not info["last_action_success"]:
                    break
        env.save_episode_log()
        if attempt == 0:
            holder["episode_info"] = {
                "planner_output_error": planner.output_json_error,
                "num_invalid_actions": episode_info["num_invalid_actions"],
                "episode_elapsed_seconds": time.time() - env._episode_start_time,
            }
        return self.finish_trace(trace, info, empty_plan)
