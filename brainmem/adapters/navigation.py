"""EB-Navigation adapter (AI2-THOR 5.0.0, atomic movement actions, 20-step budget)."""
from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Tuple

from embodiedbench.evaluator.eb_navigation_evaluator import EB_NavigationEvaluator

from brainmem.adapters.base import MAX_CONSECUTIVE_PLANNER_ERRORS, BrainMemEvaluatorMixin
from brainmem.adapters.common import repeated_failures, repeated_identical, room_type_from_floorplan
from brainmem.memory.working_memory import WorkingMemoryEntry
from brainmem.retry import RolloutTrace
from brainmem.types import StepObservation, TaskQuery
from brainmem.utils import canonical_object

_HEADINGS = {0: "north", 90: "east", 180: "south", 270: "west"}
_OPPOSITE = {"forward": "backward", "backward": "forward", "rightward": "leftward", "leftward": "rightward"}
NEAR_DISTANCE_M = 1.0      # objects closer than this are related by "near"
ALIGN_TOLERANCE_M = 0.25   # lateral offset under which two objects are "aligned with" each other


def _move_direction(action: str) -> str:
    a = action.lower()
    for d in _OPPOSITE:
        if d in a:
            return d
    return ""


def navigation_consistency(window: List[WorkingMemoryEntry], state: str) -> List[str]:
    warnings: List[str] = []
    # Oscillation: a move immediately undone by the opposite move.
    for prev, cur in zip(window, window[1:]):
        d0, d1 = _move_direction(prev.action), _move_direction(cur.action)
        if d0 and d1 and _OPPOSITE[d0] == d1:
            warnings.append(f"Steps {prev.t}-{cur.t} moved {d0} and then {d1}, returning to the same place.")
    # Revisits: the same pose reached more than once in the window.
    poses = [e.location for e in window]
    revisited = sorted({p for p in poses if poses.count(p) >= 2})
    for p in revisited:
        warnings.append(f"Position {p} was visited {poses.count(p)} times in the last {len(window)} steps; "
                        "explore a direction that has not been tried.")
    warnings.extend(repeated_failures(window))
    warnings.extend(repeated_identical(window, min_run=4))
    return warnings


class BrainMemNavigationEvaluator(BrainMemEvaluatorMixin, EB_NavigationEvaluator):
    benchmark = "eb-nav"

    def consistency_checker(self):
        return navigation_consistency

    def episode_index(self) -> int:
        if len(self.env.selected_indexes):
            return self.env.selected_indexes[self.env._current_episode_num - 1] + 1
        return self.env._current_episode_num

    def rewind_episode(self) -> None:
        self.env._current_episode_num -= 1

    # ------------------------------------------------------------------ task / observation
    def task_query(self) -> TaskQuery:
        ep = self.env.episode_data
        scene = ep["scene"]
        return TaskQuery(instruction=self.env.episode_language_instruction, task_type="navigate",
                         target_object=canonical_object(ep["targetObjectType"]), room_id=scene, scene_id=scene,
                         episode_key=f"{self.eval_set}#{self.env._current_episode_num}")

    @staticmethod
    def _agent_frame(agent: Dict[str, Any], pos: Dict[str, float]) -> Tuple[float, float]:
        """(lateral, forward) offset of ``pos`` in the agent frame (x right, z forward)."""
        yaw = math.radians(agent["rotation"]["y"])
        dx = pos["x"] - agent["position"]["x"]
        dz = pos["z"] - agent["position"]["z"]
        lateral = dx * math.cos(yaw) - dz * math.sin(yaw)
        forward = dx * math.sin(yaw) + dz * math.cos(yaw)
        return lateral, forward

    def observe(self, obs, info=None, action=None, success=None) -> StepObservation:
        meta = self.env.env.last_event.metadata
        scene = self.env.episode_data["scene"]
        agent = meta["agent"]
        x, z = agent["position"]["x"], agent["position"]["z"]
        heading = _HEADINGS.get(int(round(agent["rotation"]["y"] / 90.0)) * 90 % 360, "north")
        visible = [o for o in meta.get("objects", []) if o.get("visible")]
        names = sorted({canonical_object(o["objectType"]) for o in visible})
        relations = []
        frame = [(canonical_object(o["objectType"]), self._agent_frame(agent, o["position"]), o["position"])
                 for o in visible]
        for i in range(len(frame)):
            for j in range(i + 1, len(frame)):
                (a, (la, _), pa), (b, (lb, _), pb) = frame[i], frame[j]
                if a == b:
                    continue
                if math.hypot(pa["x"] - pb["x"], pa["z"] - pb["z"]) <= NEAR_DISTANCE_M:
                    relations.append((a, "near", b))
                    if abs(la - lb) <= ALIGN_TOLERANCE_M:
                        relations.append((a, "aligned with", b))
                    elif la < lb:
                        relations.append((a, "left of", b))
                    else:
                        relations.append((b, "left of", a))
        location = f"({x:.2f}, {z:.2f}) facing {heading}"
        state = f"standing, camera horizon {int(round(agent.get('cameraHorizon', 0)))} deg"
        summary = f"{room_type_from_floorplan(scene)} at {location}; visible: {', '.join(names[:12]) or 'nothing'}"
        tokens = [f"x:{round(x / 0.25)}", f"z:{round(z / 0.25)}", f"h:{heading}"] + names
        return StepObservation(room_id=scene, scene_id=scene, summary=summary, state_tokens=tokens,
                               agent_state=state, location=location, objects=names,
                               relations=sorted(set(relations)))

    # ------------------------------------------------------------------ rollout (mirrors EB_NavigationEvaluator.evaluate)
    def run_rollout(self, attempt: int, holder: Dict[str, Any], previous: List[RolloutTrace]) -> RolloutTrace:
        env, planner = self.env, self.planner
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
        info: Dict[str, Any] = {"task_success": 0, "env_step": 0, "episode_elapsed_seconds": 0}
        done, errors = False, 0
        while not done:
            try:
                action, reasoning = self.plan(query, planner.act, img_path, user_instruction)
                reasoning = json.loads(reasoning)
                errors = 0
            except Exception as e:
                errors += 1
                print(f"planner error ({errors}/{MAX_CONSECUTIVE_PLANNER_ERRORS}): {e}")
                if errors >= MAX_CONSECUTIVE_PLANNER_ERRORS:
                    raise
                time.sleep(1)
                continue
            actions = action if isinstance(action, list) else [action]
            limit = min(env._max_episode_steps - env._current_step + 1, len(actions)) if isinstance(action, list) else 1
            for i, a in enumerate(actions[:limit]):
                obs, reward, done, info = self.env_step(a, reasoning, 1 if i == 0 else 0)
                planner.update_info(info)
                img_path = env.save_image(obs)
                self.record_step(trace, obs, info, info["action_description"], reasoning)
                if done:
                    break
                if info["last_action_success"] == 0:
                    break
        if attempt == 0:
            holder["episode_info"] = {"planner_output_error": planner.output_json_error,
                                      "episode_elapsed_seconds": info.get("episode_elapsed_seconds", 0)}
        return self.finish_trace(trace, info, empty_plan=False)
