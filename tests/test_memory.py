"""Offline tests of the memory stores (no simulator, no API).

    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainmem.config import PRESETS, BrainMemConfig, build_config  # noqa: E402
from brainmem.memory.semantic_memory import SemanticMemory, confidence, usage, utility  # noqa: E402
from brainmem.memory.spatial_kg import SpatialKG  # noqa: E402
from brainmem.memory.trajectory_kg import TrajectoryKG  # noqa: E402
from brainmem.memory.working_memory import WorkingMemory, WorkingMemoryEntry  # noqa: E402
from brainmem.retry import RolloutTrace, expected_retry_bound, extract_failure_reasons  # noqa: E402
from brainmem.types import StepRecord  # noqa: E402


class TestUtility(unittest.TestCase):
    def test_definitions(self):
        self.assertAlmostEqual(confidence(8, 10), 0.8)
        self.assertAlmostEqual(usage(8), 0.8)
        self.assertAlmostEqual(usage(25), 1.0)
        self.assertAlmostEqual(utility(8, 10, 0.7, 0.3), 0.7 * 0.8 + 0.3 * 0.8)

    def test_paper_example_and_crossing(self):
        # 8/10 scores 0.8 for every alpha; 1/1 scores 0.9 alpha + 0.1; they cross at alpha = 7/9.
        for alpha in (0.1, 0.5, 0.7, 0.9):
            self.assertAlmostEqual(utility(8, 10, alpha, 1 - alpha), 0.8)
            self.assertAlmostEqual(utility(1, 1, alpha, 1 - alpha), 0.9 * alpha + 0.1)
        cross = 7 / 9
        self.assertAlmostEqual(utility(1, 1, cross, 1 - cross), 0.8)
        self.assertAlmostEqual(cross, 0.78, places=2)
        # alpha = 0.7 lies below the crossing: the well-supported guideline wins.
        self.assertGreater(utility(8, 10, 0.7, 0.3), utility(1, 1, 0.7, 0.3))
        self.assertLess(utility(8, 10, 0.9, 0.1), utility(1, 1, 0.9, 0.1))

    def test_retry_bound(self):
        self.assertAlmostEqual(expected_retry_bound(3, 0.82), 0.54)


class TestSemanticMemory(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.sem = SemanticMemory(self.dir, alpha=0.7, beta=0.3, max_guidelines=20, prune_trigger=30)

    def test_merge_and_counts(self):
        g1 = self.sem.add_or_merge_guideline("Before turning on the faucet, place the object in the sink.",
                                             ["pick_clean_then_place_in_recep"], ["fork"], "exp_0")
        g2 = self.sem.add_or_merge_guideline("Place the object in the sink before turning on the faucet.",
                                             ["pick_clean_then_place_in_recep"], ["ladle"], "exp_1")
        self.assertEqual(g1["guideline_id"], g2["guideline_id"])
        self.assertEqual(len(self.sem.guidelines), 1)
        self.assertEqual((g1["n_success"], g1["n_total"]), (1, 1))
        self.sem.record_application([g1["guideline_id"]], success=False)
        self.assertEqual((g1["n_success"], g1["n_total"]), (1, 2))
        self.assertAlmostEqual(g1["utility"], 0.7 * 0.5 + 0.3 * 0.1)

    def test_pruning_with_protection(self):
        protected = self.sem.add_or_merge_guideline("protected rule alpha beta", ["t"], [], "exp_p")
        for _ in range(9):
            self.sem.record_application([protected["guideline_id"]], success=True)  # 10/10
        for i in range(30):
            g = self.sem.add_or_merge_guideline(f"rule number {i} unique token{i} x{i} y{i}", ["t"], [], f"exp_{i}")
            self.sem.record_application([g["guideline_id"]], success=False)  # 1/2
        self.assertEqual(len(self.sem.guidelines), 31)
        removed = self.sem.prune()
        self.assertEqual(len(self.sem.guidelines), 20)
        self.assertEqual(len(removed), 11)
        self.assertIn(protected["guideline_id"], self.sem.guidelines)
        self.assertEqual(self.sem.prune(), [])  # at or below 1.5 M nothing happens

    def test_retrieval_top_k(self):
        a = self.sem.add_or_merge_guideline("rule a about knives", ["t"], ["knife"], "e0")
        b = self.sem.add_or_merge_guideline("rule b about forks", ["t"], ["fork"], "e1")
        self.sem.add_or_merge_guideline("rule c about mugs", ["other"], ["mug"], "e2")
        got = self.sem.retrieve_guidelines("t", "fork", k=2)
        self.assertEqual([g["guideline_id"] for g in got], [b["guideline_id"], a["guideline_id"]])

    def test_persistence(self):
        self.sem.add_or_merge_guideline("persist me", ["t"], [], "e0")
        self.sem.add_record({"episode_key": "ep", "instruction": "x", "room_id": "r", "task_type": "t",
                             "target_object": "o", "success": True})
        self.sem.save()
        again = SemanticMemory(self.dir)
        self.assertEqual(len(again.guidelines), 1)
        self.assertEqual(len(again.records), 1)


class TestTrajectoryKG(unittest.TestCase):
    def test_merge_rules_and_recall(self):
        d = tempfile.mkdtemp()
        kg = TrajectoryKG(d, state_merge_threshold=0.8)
        for success in (True, True, False):
            kg.start_rollout("ep", 0, "clean", "fork", "Rinse the fork and put it on the table", "kitchen",
                             "visible: fork, sink", ["fork", "sink"])
            kg.add_step("find a Fork", True, "kitchen", "near fork", ["fork", "sink", "near:fork"])
            if success:
                kg.add_step("pick up the Fork", True, "kitchen", "holding fork", ["fork", "sink", "hold:fork"])
            else:
                kg.add_step("turn on the Faucet", False, "kitchen", "near fork", ["fork", "sink", "near:fork"],
                            feedback="Last action is invalid.")
            kg.end_rollout(success)
        # state deduplication: s0, near-fork, holding-fork
        self.assertEqual(len(kg.states), 3)
        # transition aggregation: find (x3), pick (x2), faucet (x1)
        counts = sorted(e["count"] for e in kg.edges.values())
        self.assertEqual(counts, [1, 2, 3])
        seqs = kg.recall_sequences("clean", "fork", "Rinse a fork", n=2)
        self.assertEqual(seqs[0], (["find a Fork", "pick up the Fork"], 2))
        self.assertEqual(len(seqs), 1)  # the failed rollout is not recalled
        failed = kg.failed_transitions("clean", "kitchen", None, n=3)
        self.assertEqual([f["action"] for f in failed], ["turn on the Faucet"])
        self.assertEqual(kg.failed_transitions("heat", "kitchen", None, n=3), [])
        kg.save()
        self.assertEqual(TrajectoryKG(d).stats(), kg.stats())


class TestSpatialKG(unittest.TestCase):
    def test_support_counted_once_per_episode(self):
        kg = SpatialKG(tempfile.mkdtemp(), stable_support=2)
        for ep in ("e1", "e1", "e2"):  # e1 appears twice (a retry of the same episode)
            kg.begin_episode(ep)
            kg.observe("FloorPlan1", "FloorPlan1", ["fork"], [("fork", "on", "countertop")])
            kg.observe("FloorPlan1", "FloorPlan1", ["fork"], [("fork", "on", "countertop")])
            kg.consolidate()
        rel = kg.retrieve("FloorPlan1", "FloorPlan1", n=5)["relations"][0]
        self.assertEqual(rel["support"], 2)
        self.assertTrue(rel["stable"])


class TestWorkingMemory(unittest.TestCase):
    def test_window_and_contents(self):
        wm = WorkingMemory(3, consistency_checker=lambda w, s: ["warn"] if len(w) == 3 else [])
        wm.clear("holding nothing", "kitchen")
        for t in range(1, 6):
            wm.append(WorkingMemoryEntry(f"a{t}", "ok", t % 2 == 1, "holding nothing", "kitchen", t))
        self.assertEqual([e.t for e in wm.entries], [3, 4, 5])
        text = wm.render()
        self.assertIn("Cumulative action success rate: 3/5", text)
        self.assertIn("Consistency warnings", text)
        wm2 = WorkingMemory(3, include_feedback=False, include_agent_state=False, include_consistency_check=False)
        wm2.append(WorkingMemoryEntry("a1", "", None, "holding fork", "kitchen", 1))
        text2 = wm2.render()
        self.assertNotIn("Feedback", text2)
        self.assertNotIn("success rate", text2)
        self.assertNotIn("holding fork", text2)


class TestRetry(unittest.TestCase):
    def test_failure_reasons(self):
        tr = RolloutTrace(attempt=0, termination="step_budget")
        tr.steps = [StepRecord(1, "turn on the Faucet", "Last action is invalid.", False)] + \
                   [StepRecord(i, "find a Sink", "ok", True) for i in range(2, 5)]
        reasons = extract_failure_reasons(tr, max_steps=30)
        self.assertTrue(reasons[0].startswith("'turn on the Faucet' failed at step 1"))
        self.assertTrue(any("executed 3 times" in r for r in reasons))
        self.assertTrue(any("30 environment steps" in r for r in reasons))


class TestConfig(unittest.TestCase):
    def test_defaults_match_paper(self):
        c = BrainMemConfig()
        self.assertEqual((c.window_size, c.top_k, c.alpha, c.beta, c.max_guidelines, c.prune_trigger,
                          c.protect_confidence, c.protect_min_success, c.max_retries),
                         (5, 2, 0.7, 0.3, 20, 30, 0.8, 5, 3))

    def test_presets_table_viii(self):
        rows = {"base": (False, False, False, False), "wm": (True, False, False, False),
                "wm_fb": (True, False, False, True), "wm_em": (True, True, False, False),
                "wm_em_fb": (True, True, False, True), "wm_em_sm": (True, True, True, False),
                "full": (True, True, True, True)}
        for name, (wm, em, sm, fb) in rows.items():
            c = build_config(name)
            if not c.enabled:
                self.assertEqual(name, "base")
                continue
            self.assertEqual((c.use_working_memory, c.use_episodic_memory, c.use_semantic_memory, c.use_feedback),
                             (wm, em, sm, fb), name)
        self.assertTrue(set(rows) <= set(PRESETS))

    def test_alpha_beta_constraint(self):
        with self.assertRaises(ValueError):
            build_config("full", {"alpha": 0.6, "beta": 0.3})


if __name__ == "__main__":
    unittest.main()
