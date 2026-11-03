"""Aggregate BrainMem / baseline runs into the metrics reported in the paper.

    python tools/summarize.py running/eb_alfred/gpt-4o_brainmem_full [more run dirs ...] [--json out.json]

For every eval set found below a run directory it reports

  SR / SSR            task success and subgoal success (task_progress) of the first attempt tau(0)
  Plan / Env / Total  average planner steps, environment steps and their sum per episode (tau(0))
  retries/episode     average number of guided-retry rollouts, with the bound R * (1 - SR_0)
  retry share         retry rollouts / all rollouts, and their share of wall-clock time
  tokens              provider-reported tokens of tau(0) + consolidation, and of retries
  latency             per-decision latency of tau(0), retrieval and step-update latency
  wall clock          total time and its decomposition (planner, simulation, consolidation,
                      retrieval + updates)
  memory              storage size and graph sizes after the last episode
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def load_episodes(run_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    by_set: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in glob.glob(os.path.join(run_dir, "**", "brainmem", "episode_*.json"), recursive=True):
        with open(path) as f:
            ep = json.load(f)
        by_set[ep["eval_set"]].append(ep)
    for eps in by_set.values():
        eps.sort(key=lambda e: e["timestamp"])
    return by_set


def _phase_seconds(ep: Dict[str, Any], category: str) -> float:
    return sum(v.get(category, 0.0) for v in ep["profile"]["seconds"].values())


def summarize(eps: List[Dict[str, Any]], max_retries: int) -> Dict[str, Any]:
    first = [e["rollouts"][0] for e in eps]
    n = len(eps)
    sr0 = float(np.mean([r["success"] for r in first]))
    retries = [len(e["rollouts"]) - 1 for e in eps]
    tok = [e["profile"]["tokens"] for e in eps]

    def tsum(phase):
        return int(sum(t[phase]["input_tokens"] + t[phase]["output_tokens"] for t in tok))

    wall_total = sum(sum(sum(v.values()) for v in e["profile"]["seconds"].values()) for e in eps)
    retry_seconds = sum(sum(e["profile"]["seconds"]["retry"].values()) for e in eps)
    decisions = [x for e in eps for x in e["profile"]["decision_latency_s"]["attempt_0"]]
    retrieval = [x for e in eps for x in e["profile"]["retrieval_latency_s"]]
    updates = [x for e in eps for x in e["profile"]["update_latency_s"]]
    last_mem = eps[-1].get("memory_stats", {}) if eps else {}
    half = n // 2
    out = {
        "episodes": n,
        "SR": 100 * sr0,
        "SSR": 100 * float(np.mean([r["task_progress"] for r in first])),
        "plan_steps": float(np.mean([r["planner_steps"] for r in first])),
        "env_steps": float(np.mean([r["env_steps"] for r in first])),
        "final_SR_after_retry": 100 * float(np.mean([any(r["success"] for r in e["rollouts"]) for e in eps])),
        "retries_per_episode": float(np.mean(retries)),
        "retry_bound_R(1-SR0)": max_retries * (1 - sr0),
        "retry_rollouts": int(sum(retries)),
        "total_rollouts": int(n + sum(retries)),
        "retry_rollout_share": float(sum(retries) / (n + sum(retries))) if n else 0.0,
        "tokens_attempt0_plus_consolidation": tsum("attempt_0") + tsum("consolidation"),
        "tokens_attempt0": tsum("attempt_0"),
        "tokens_consolidation": tsum("consolidation"),
        "tokens_retry": tsum("retry"),
        "decision_latency_s_mean": float(np.mean(decisions)) if decisions else 0.0,
        "retrieval_latency_ms_mean": 1000 * float(np.mean(retrieval)) if retrieval else 0.0,
        "update_latency_ms_mean": 1000 * float(np.mean(updates)) if updates else 0.0,
        "consolidation_s_mean": float(np.mean([e["profile"]["seconds"]["consolidation"].get("consolidation", 0.0)
                                               for e in eps])),
        "wall_min_total": wall_total / 60,
        "wall_min_planner": sum(_phase_seconds(e, "planner_inference") for e in eps) / 60,
        "wall_min_env": sum(_phase_seconds(e, "env_simulation") for e in eps) / 60,
        "wall_min_consolidation": sum(_phase_seconds(e, "consolidation") for e in eps) / 60,
        "wall_min_retrieval_and_updates": sum(_phase_seconds(e, "retrieval") + _phase_seconds(e, "step_update")
                                              for e in eps) / 60,
        "wall_min_retry": retry_seconds / 60,
        "SR_first_half": 100 * float(np.mean([r["success"] for r in first[:half]])) if half else 0.0,
        "SR_second_half": 100 * float(np.mean([r["success"] for r in first[half:]])) if n - half else 0.0,
    }
    out["plan_plus_env_steps"] = out["plan_steps"] + out["env_steps"]
    if retrieval:
        k = max(1, len(retrieval) // 10)
        out["retrieval_latency_ms_first_decile"] = 1000 * float(np.mean(retrieval[:k]))
        out["retrieval_latency_ms_last_decile"] = 1000 * float(np.mean(retrieval[-k:]))
    out.update({f"memory_{k}": v for k, v in last_mem.items()})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--json", default="")
    args = p.parse_args()
    report: Dict[str, Any] = {}
    for run_dir in args.run_dirs:
        report[run_dir] = {s: summarize(eps, int(eps[0]["max_retries"]))
                           for s, eps in sorted(load_episodes(run_dir).items())}
    for run_dir, sets in report.items():
        print(f"\n=== {run_dir}")
        keys = ["episodes", "SR", "SSR", "plan_steps", "env_steps", "plan_plus_env_steps", "retries_per_episode",
                "retry_bound_R(1-SR0)", "tokens_attempt0_plus_consolidation", "decision_latency_s_mean",
                "wall_min_total"]
        print("eval_set".ljust(22) + "".join(k[:14].rjust(16) for k in keys))
        for s, m in sets.items():
            print(s.ljust(22) + "".join(f"{m[k]:16.2f}" if isinstance(m[k], float) else f"{m[k]:16d}" for k in keys))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
