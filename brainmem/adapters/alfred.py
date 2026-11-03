"""EB-ALFRED adapter (AI2-THOR 2.1.0, high-level skills, 30-step budget)."""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

from embodiedbench.evaluator.eb_alfred_evaluator import EB_AlfredEvaluator

from brainmem.adapters.base import MAX_CONSECUTIVE_PLANNER_ERRORS, BrainMemEvaluatorMixin
from brainmem.adapters.common import count_recent, repeated_failures, room_type_from_floorplan
from brainmem.memory.working_memory import WorkingMemoryEntry
from brainmem.retry import RolloutTrace
from brainmem.types import StepObservation, TaskQuery
from brainmem.utils import canonical_object

_CONTAINERS = ("sink", "basin", "drawer", "cabinet", "fridge", "microwave", "box", "bowl", "mug", "cup",
               "pot", "pan", "safe", "garbagecan", "bathtub")


def _support_predicate(parent: str) -> str:
    return "in" if any(c in parent for c in _CONTAINERS) else "on"


def alfred_consistency(window: List[WorkingMemoryEntry], state: str) -> List[str]:
    """Invalid action sequences detected in the working-memory window."""
    warnings: List[str] = []
    for e in window:
        a = e.action.lower()
        if "turn on" in a and "faucet" in a and e.state.startswith("holding ") and e.state != "holding nothing":
            held = e.state[len("holding "):]
            warnings.append(f"Step {e.t} turned on the faucet while still holding {held}; "
                            f"put {held} in the sink before turning on the faucet.")
        if e.success is False and "pick up" in a and e.state not in ("", "holding nothing"):
            warnings.append(f"Step {e.t} tried to pick up an object while {e.state}; put it down first.")
        if e.success is False and "put down" in a and e.state == "holding nothing":
            warnings.append(f"Step {e.t} tried to put down an object while holding nothing.")
    toggles = [e for e in window if re.match(r"turn (on|off) ", e.action.lower())]
    if len(toggles) >= 3:
        target = toggles[-1].action.split(" ", 3)[-1]
        warnings.append(f"{target} was toggled {len(toggles)} times in the last {len(window)} steps "
                        "without progress; check the preconditions of the step you are trying to complete.")
    for action, n in count_recent(window, prefix="find a ", min_count=3):
        warnings.append(f"'{action}' was executed {n} times in the last {len(window)} steps; "
                        "search another location or open a closed receptacle.")
    warnings.extend(repeated_failures(window))
    return warnings


class BrainMemAlfredEvaluator(BrainMemEvaluatorMixin, EB_AlfredEvaluator):
    benchmark = "eb-alf"

    def __init__(self, config):
        super().__init__(config)
        self.episode_order_seed = config.get("episode_order_seed")
        self._near = ""

    # ------------------------------------------------------------------ eval-set hooks
    def prepare_eval_set(self) -> None:
        if self.episode_order_seed is not None:
            n = len(self.env.dataset)
            perm = np.random.RandomState(int(self.episode_order_seed)).permutation(n).tolist()
            base = self.env.selected_indexes if len(self.env.selected_indexes) else list(range(n))
            self.env.dataset = [self.env.dataset[i] for i in perm]
            self.env.selected_indexes = [base[i] for i in perm]

    def consistency_checker(self):
        return alfred_consistency

    def episode_index(self) -> int:
        if len(self.env.selected_indexes):
            return self.env.selected_indexes[self.env._current_episode_num - 1] + 1
        return self.env._current_episode_num

    def rewind_episode(self) -> None:
        self.env._current_episode_num -= 1

    # ------------------------------------------------------------------ task / observation
    def task_query(self) -> TaskQuery:
        entry = self.env.dataset[self.env._current_episode_num - 1]
        task_dir = entry["task"].split("/")[0]  # e.g. pick_clean_then_place_in_recep-Ladle-None-DiningTable-4
        fields = task_dir.split("-")
        scene = self.env.env.last_event.metadata.get("sceneName", "")
        return TaskQuery(instruction=self.env.episode_language_instruction, task_type=fields[0],
                         target_object=canonical_object(fields[1]) if len(fields) > 1 else "",
                         room_id=scene, scene_id=scene,
                         episode_key=f"{self.eval_set}/{entry['task']}#{entry['repeat_idx']}")

    def _name(self, object_id: str) -> str:
        id_to_name = self.env.id_to_name_dict or {}
        return canonical_object(id_to_name.get(object_id, object_id))

    def observe(self, obs, info=None, action=None, success=None) -> StepObservation:
        meta = self.env.env.last_event.metadata
        scene = meta.get("sceneName", "")
        room = room_type_from_floorplan(scene)
        if action and success and action.lower().startswith("find a"):
            self._near = action.split("find a", 1)[1].strip()
        inventory = meta.get("inventoryObjects") or []
        held = canonical_object(inventory[0]["objectType"]) if inventory else ""
        visible, relations = [], []
        for o in meta.get("objects", []):
            if not o.get("visible"):
                continue
            name = canonical_object(o["objectType"])
            visible.append(name)
            for pid in o.get("parentReceptacles") or []:
                parent = self._name(pid)
                if parent and parent != name:
                    relations.append((name, _support_predicate(parent), parent))
        visible = sorted(set(visible))
        state = f"holding {held}" if held else "holding nothing"
        near = self._near.lower()
        location = f"{room} (near {self._near})" if self._near else room
        rel_text = "; ".join(f"{a} {p} {b}" for a, p, b in sorted(set(relations))[:6])
        summary = f"{location}; {state}; visible: {', '.join(visible[:12]) or 'nothing'}" + \
                  (f"; {rel_text}" if rel_text else "")
        state_tokens = visible + [f"near:{near or 'none'}", f"hold:{held or 'nothing'}"]
        return StepObservation(room_id=scene, scene_id=scene, summary=summary, state_tokens=state_tokens,
                               agent_state=state, location=location, objects=visible,
                               relations=sorted(set(relations)))

    # ------------------------------------------------------------------ rollout (mirrors EB_AlfredEvaluator.evaluate)
    def run_rollout(self, attempt: int, holder: Dict[str, Any], previous: List[RolloutTrace]) -> RolloutTrace:
        env, planner = self.env, self.planner
        episode_info = {"reward": [], "num_invalid_actions": 0, "empty_plan": 0}
        self._near = ""
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
        planner.set_actions(env.language_skill_set)
        trace = RolloutTrace(attempt=attempt)
        info: Dict[str, Any] = {"task_success": 0, "task_progress": 0, "env_step": 0}
        done, empty_plan, errors = False, False, 0
        while not done:
            try:
                action, reasoning = self.plan(query, planner.act, img_path, user_instruction)
                errors = 0
            except Exception as e:  # provider errors; EmbodiedBench waits and retries
                errors += 1
                print(f"planner error ({errors}/{MAX_CONSECUTIVE_PLANNER_ERRORS}): {e}")
                if errors >= MAX_CONSECUTIVE_PLANNER_ERRORS:
                    raise
                time.sleep(30)
                continue
            if action == -2:  # empty plan
                episode_info["empty_plan"] = 1
                empty_plan = True
                env.episode_log.append({"last_action_success": 0.0, "action_id": -2,
                                        "action_description": "empty plan", "reasoning": reasoning})
                break
            if action == -1:  # invalid planner output
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
                action_str = a if isinstance(a, str) else env.language_skill_set[a]
                planner.update_info(info)
                img_path = env.save_image(obs)
                episode_info["reward"].append(reward)
                episode_info["num_invalid_actions"] += (info["last_action_success"] == 0)
                self.record_step(trace, obs, info, action_str, reasoning)
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
