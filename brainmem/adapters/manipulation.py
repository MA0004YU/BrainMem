"""EB-Manipulation adapter (CoppeliaSim/PyRep, 7-D discrete gripper actions, 15-step budget)."""
from __future__ import annotations

import ast
import copy
import time
from typing import Any, Dict, List, Optional

from embodiedbench.envs.eb_manipulation.eb_man_utils import (draw_bounding_boxes, draw_xyz_coordinate,
                                                              form_object_coord_for_input)
from embodiedbench.evaluator.eb_manipulation_evaluator import EB_ManipulationEvaluator

from brainmem.adapters.base import BrainMemEvaluatorMixin
from brainmem.adapters.common import repeated_failures, repeated_identical
from brainmem.memory.working_memory import WorkingMemoryEntry
from brainmem.retry import RolloutTrace
from brainmem.types import StepObservation, TaskQuery

ALIGN_TOLERANCE = 3   # voxels: offsets within this are treated as equal
STATE_BUCKET = 10     # voxels: gripper positions are bucketed for state deduplication


def _parse_action(action: str) -> Optional[List[int]]:
    try:
        v = ast.literal_eval(action)
    except (ValueError, SyntaxError):
        return None
    return list(v) if isinstance(v, (list, tuple)) and len(v) == 7 else None


def manipulation_consistency(window: List[WorkingMemoryEntry], state: str) -> List[str]:
    warnings: List[str] = []
    prev = None
    for e in window:
        v = _parse_action(e.action)
        if v is not None and prev is not None and v[6] == 0 and prev[6] == 0 and v[:3] == prev[:3]:
            warnings.append(f"Step {e.t} closed the gripper again at the same position {v[:3]} without moving.")
        prev = v if v is not None else prev
    warnings.extend(repeated_failures(window))
    warnings.extend(repeated_identical(window, min_run=2))
    return warnings


def coordinate_relations(coords: Dict[str, List[int]]):
    """Directional and alignment relations between objects in the table frame."""
    rels = []
    names = sorted(coords)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = coords[a], coords[b]
            dx, dy, dz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
            if abs(dx) <= ALIGN_TOLERANCE and abs(dy) <= ALIGN_TOLERANCE:
                if dz > ALIGN_TOLERANCE:
                    rels.append((b, "above", a))
                elif dz < -ALIGN_TOLERANCE:
                    rels.append((a, "above", b))
                continue
            if abs(dy) <= ALIGN_TOLERANCE:
                rels.append((a, "aligned along X with", b))
            elif abs(dx) <= ALIGN_TOLERANCE:
                rels.append((a, "aligned along Y with", b))
            if abs(dx) > ALIGN_TOLERANCE:
                rels.append((b, "in +X direction of", a) if dx > 0 else (a, "in +X direction of", b))
            if abs(dy) > ALIGN_TOLERANCE:
                rels.append((b, "in +Y direction of", a) if dy > 0 else (a, "in +Y direction of", b))
    return rels


class BrainMemManipulationEvaluator(BrainMemEvaluatorMixin, EB_ManipulationEvaluator):
    benchmark = "eb-man"

    def __init__(self, config):
        super().__init__(config)
        self._coords: Dict[str, List[int]] = {}
        self._gripper: Optional[List[int]] = None

    def consistency_checker(self):
        return manipulation_consistency

    def episode_index(self) -> int:
        return self.env._current_episode_num

    def rewind_episode(self) -> None:
        self.env._current_episode_num -= 1

    def finish_eval_set(self) -> None:
        # EB_ManipulationEvaluator.evaluate ends with these two calls.
        self.print_task_eval_results(filename="summary.json")
        self.env.close()

    # ------------------------------------------------------------------ task / observation
    def task_query(self) -> TaskQuery:
        variation = self.env.current_task_variation
        return TaskQuery(instruction=self.env.episode_language_instruction, task_type=variation,
                         target_object=self.env.task_class, room_id=variation,
                         scene_id=f"{variation}#{self.env._current_episode_num}",
                         episode_key=f"{self.eval_set}#{self.env._current_episode_num}")

    def observe(self, obs, info=None, action=None, success=None) -> StepObservation:
        variation = self.env.current_task_variation
        scene = f"{variation}#{self.env._current_episode_num}"
        if action is not None:
            v = _parse_action(action)
            if v is not None and success:
                self._gripper = v
        if self._gripper is None:
            state, location, tokens = "gripper open", "initial pose", ["g:init", "open"]
        else:
            g = self._gripper
            state = "gripper open" if g[6] == 1 else "gripper closed"
            location = f"gripper at [{g[0]}, {g[1]}, {g[2]}]"
            tokens = [f"gx:{g[0] // STATE_BUCKET}", f"gy:{g[1] // STATE_BUCKET}", f"gz:{g[2] // STATE_BUCKET}",
                      "open" if g[6] == 1 else "closed"]
        objs = ", ".join(f"{k} {v}" for k, v in sorted(self._coords.items()))
        summary = f"{state}; {location}; objects: {objs or 'unknown'}"
        return StepObservation(room_id=variation, scene_id=scene, summary=summary, state_tokens=tokens,
                               agent_state=state, location=location, objects=sorted(self._coords),
                               relations=coordinate_relations(self._coords))

    def _object_coords(self, obs, camera_views):
        avg_obj_coord, all_avg_point_list, extr, intr = form_object_coord_for_input(
            obs, self.env.task_class, camera_views)
        self._coords = {k: list(map(int, v)) for k, v in avg_obj_coord.items()}
        return avg_obj_coord, all_avg_point_list, extr, intr

    def _decorate(self, img_path_list, all_avg_point_list, extr, intr):
        if not self.config["language_only"]:
            for i, p in enumerate(img_path_list):
                if "front_rgb" in p:
                    img_path_list[i] = draw_xyz_coordinate(p, self.config["resolution"])
        if self.config["detection_box"] and not self.config["language_only"]:
            img_path_list = draw_bounding_boxes(img_path_list, all_avg_point_list, extr, intr)
        return img_path_list

    # ------------------------------------------------------------------ rollout (mirrors EB_ManipulationEvaluator.evaluate)
    def run_rollout(self, attempt: int, holder: Dict[str, Any], previous: List[RolloutTrace]) -> RolloutTrace:
        env, planner, cfg = self.env, self.planner, self.config
        self._gripper = None
        image_history: List[str] = []
        _, obs = self.env_reset()
        camera_views = ["front_rgb", "wrist_rgb"] if cfg["multiview"] else ["front_rgb"]
        img_path_list = env.save_image(camera_views)
        avg_obj_coord, pts, extr, intr = self._object_coords(vars(copy.deepcopy(obs)), camera_views)
        img_path_list = self._decorate(img_path_list, pts, extr, intr)
        if cfg["multistep"]:
            image_history.append(img_path_list[0])
        user_instruction = env.episode_language_instruction
        query = self.task_query()
        if attempt == 0:
            holder["query"] = query
            if self.bm is not None:
                self.bm.begin_episode(query)
        self.begin_rollout(query, obs, attempt, previous)
        planner.reset()
        trace = RolloutTrace(attempt=attempt)
        info: Dict[str, Any] = {"task_success": 0, "episode_elapsed_seconds": 0}
        done, empty_plan = False, False
        reasoning_list: List[str] = []
        while not done:
            images = image_history if cfg["multistep"] else img_path_list
            action, reasoning = self.plan(query, planner.act, images, user_instruction, str(avg_obj_coord),
                                          env.current_task_variation)
            reasoning_list.append(reasoning)
            if len(action) == 0:
                info = {"task_success": 0, "episode_elapsed_seconds": 0}
                empty_plan = True
                break
            for a in action[:min(env._max_episode_steps - env._current_step, len(action))]:
                obs, reward, done, info = self.env_step(a)
                planner.update_info(info)
                img_path_list = env.save_image(camera_views)
                if cfg["multistep"]:
                    image_history.extend(img_path_list)
                info["env_step"] = env._current_step
                self.record_step(trace, obs, info, str(list(a)), reasoning)
                if done:
                    break
            avg_obj_coord, pts, extr, intr = self._object_coords(copy.deepcopy(obs), camera_views)
            if not done:
                img_path_list = self._decorate(img_path_list, pts, extr, intr)
                if cfg["multistep"] and image_history[-1].split(".png")[0] in img_path_list[0]:
                    image_history.pop()
                    image_history.append(img_path_list[0])
        if attempt == 0:
            holder["episode_info"] = {"planner_output_error": planner.output_json_error,
                                      "episode_elapsed_seconds": info.get("episode_elapsed_seconds", 0)}
            self.save_planner_outputs(reasoning_list)
        return self.finish_trace(trace, info, empty_plan)
