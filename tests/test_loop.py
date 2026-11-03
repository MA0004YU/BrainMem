"""End-to-end test of the BrainMem episode loop with a toy environment and planner.

The toy task is solved by the action sequence ["a", "b"]. Without memory the
toy planner outputs ["a", "c"] (action "c" fails); when its prompt contains a
retry context or a guideline it outputs ["a", "b"]. The test checks that

  * the reported result is the first attempt tau(0), not the retry;
  * a failed first attempt triggers guided retry, which stops at the first success;
  * consolidation makes exactly one experience-agent call and stores a guideline;
  * the guideline is retrieved for the next episode and its counts are updated;
  * result files are written in the EmbodiedBench-compatible flat format.
"""
import glob
import json
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainmem.adapters.base import BrainMemEvaluatorMixin  # noqa: E402
from brainmem.config import build_config  # noqa: E402
from brainmem.retry import RolloutTrace  # noqa: E402
from brainmem.types import StepObservation, TaskQuery  # noqa: E402

GUIDELINE = "Execute action b right after action a."


class FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        system = kwargs["messages"][0]["content"]
        if "experience agent" in system:
            text = json.dumps({"action_sequence_summary": "did a then b", "success_pattern": "a then b",
                               "failure_reasons": ["c is invalid"], "learning_insights": ["b follows a"],
                               "guidelines": [{"text": GUIDELINE, "task_types": ["toy"], "objects": []}]})
        else:
            text = "plan"
        usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=10)
        msg = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(usage=usage, choices=[types.SimpleNamespace(message=msg)])


class FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = types.SimpleNamespace(completions=FakeCompletions(self))


class ToyPlanner:
    model_name = "gpt-4o"
    model_type = "remote"

    def __init__(self):
        self.model = types.SimpleNamespace(model=FakeClient(), model_type="remote")
        self.prompts = []
        self.reset()

    def reset(self):
        self.planner_steps = 0
        self.output_json_error = 0

    def process_prompt(self, instruction):
        return f"Instruction: {instruction}"

    def act(self, instruction):
        prompt = self.process_prompt(instruction)
        self.prompts.append(prompt)
        self.model.model.chat.completions.create(model=self.model_name,
                                                 messages=[{"role": "system", "content": "planner"},
                                                           {"role": "user", "content": prompt}])
        self.planner_steps += 1
        if "Retry context" in prompt or GUIDELINE in prompt:
            return ["a", "b"], "{}"
        return ["a", "c"], "{}"


class ToyEnv:
    def __init__(self, log_path, n):
        self.log_path = log_path
        self.number_of_episodes = n
        self._current_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 5
        self._cur_invalid_actions = 0
        self._max_invalid_actions = 10
        self.resets = 0

    def reset(self):
        self._current_episode_num += 1
        self._current_step = 0
        self.done_a = False
        self.resets += 1
        return {}

    def step(self, action):
        self._current_step += 1
        ok = action == "a" or (action == "b" and self.done_a)
        self.done_a = self.done_a or action == "a"
        success = action == "b" and ok
        info = {"last_action_success": float(ok), "env_feedback": "ok" if ok else "invalid",
                "env_step": self._current_step, "task_success": float(success),
                "task_progress": 1.0 if success else 0.5 * self.done_a}
        done = success or self._current_step >= self._max_episode_steps
        return {}, 0.0, done, info


class ToyEvaluator(BrainMemEvaluatorMixin):
    benchmark = "toy"

    def __init__(self, log_path, n):
        self.env = ToyEnv(log_path, n)
        self.planner = ToyPlanner()
        self.eval_set = "base"

    def consistency_checker(self):
        return None

    def episode_index(self):
        return self.env._current_episode_num

    def rewind_episode(self):
        self.env._current_episode_num -= 1

    def observe(self, obs, info=None, action=None, success=None):
        tokens = ["a_done"] if getattr(self.env, "done_a", False) else ["start"]
        return StepObservation("room", "room", " ".join(tokens), tokens, "idle", "room")

    def run_rollout(self, attempt, holder, previous):
        self.env_reset()
        query = TaskQuery("do the toy task", "toy", "", "room", "room", f"ep{self.env._current_episode_num}")
        if attempt == 0:
            holder["query"] = query
            if self.bm is not None:
                self.bm.begin_episode(query)
        self.begin_rollout(query, {}, attempt, previous)
        self.planner.reset()
        trace = RolloutTrace(attempt=attempt)
        info, done = {}, False
        while not done:
            actions, reasoning = self.plan(query, self.planner.act, "do the toy task")
            for a in actions:
                _, _, done, info = self.env_step(a)
                self.record_step(trace, {}, info, a, reasoning)
                if done or not info["last_action_success"]:
                    break
        if attempt == 0:
            holder["episode_info"] = {"planner_output_error": 0}
        return self.finish_trace(trace, info, empty_plan=False)


def _load(path):
    with open(path) as f:
        return json.load(f)


class TestLoop(unittest.TestCase):
    def _run(self, preset, n=2):
        root = tempfile.mkdtemp()
        ev = ToyEvaluator(os.path.join(root, "log"), n)
        ev.setup_brainmem(build_config(preset), os.path.join(root, "run"))
        ev.evaluate()
        flat = [_load(p) for p in sorted(glob.glob(os.path.join(root, "log", "results", "*.json")))]
        detail = [_load(p) for p in sorted(glob.glob(os.path.join(root, "log", "brainmem", "*.json")))]
        return ev, flat, detail

    def test_full_preset(self):
        ev, flat, detail = self._run("full")
        # episode 1: tau(0) fails, the first retry succeeds; the reported result is tau(0)
        self.assertEqual(flat[0]["task_success"], 0.0)
        self.assertEqual(flat[0]["final_success"], 1.0)
        self.assertEqual(flat[0]["num_retries"], 1)
        # episode 2: the guideline distilled from episode 1 is in the prompt -> first attempt succeeds
        self.assertEqual(flat[1]["task_success"], 1.0)
        self.assertEqual(flat[1]["num_retries"], 0)
        self.assertIn(GUIDELINE, ev.planner.prompts[-1])
        # one experience-agent call per episode
        agent_calls = [c for c in ev.planner.model.model.calls if "experience agent" in c["messages"][0]["content"]]
        self.assertEqual(len(agent_calls), 2)
        # the guideline was applied in episode 2 (success) on top of its creation count
        g = list(ev.bm.sem.guidelines.values())[0]
        self.assertEqual((g["n_success"], g["n_total"]), (2, 2))
        # flat result files contain only numbers or strings
        for f in flat:
            self.assertTrue(all(isinstance(v, (int, float, str)) for v in f.values()))
        # tokens: retries and consolidation are accounted separately
        self.assertGreater(flat[0]["tokens_retry"], 0)
        self.assertGreater(flat[0]["tokens_consolidation"], 0)
        self.assertEqual(detail[0]["max_retries"], 3)
        self.assertEqual(ev.env.resets, 3)  # tau(0) + 1 retry + episode 2

    def test_base_preset_is_plain_single_attempt(self):
        ev, flat, _ = self._run("base")
        self.assertEqual([f["task_success"] for f in flat], [0.0, 0.0])
        self.assertEqual([f["num_retries"] for f in flat], [0, 0])
        self.assertIsNone(ev.bm)
        self.assertNotIn("BrainMem", ev.planner.prompts[0])

    def test_no_feedback_disables_retry(self):
        _, flat, _ = self._run("wm_em_sm")
        self.assertEqual([f["num_retries"] for f in flat], [0, 0])


if __name__ == "__main__":
    unittest.main()
