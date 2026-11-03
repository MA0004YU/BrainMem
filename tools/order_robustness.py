"""Table XIV: robustness to the episode order.

    python tools/order_robustness.py running/eb_alfred/gpt-4o_order_seed1 ... running/eb_alfred/gpt-4o_order_seed5

For each run (one random episode order) it reports the first-attempt SR, and
the SR on the first and on the last half of the episodes in the order they
were executed. The spread across orders is the sample standard deviation.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import load_episodes  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--eval_set", default="base")
    args = p.parse_args()
    srs, gains = [], []
    print(f"{'run':50s} {'SR':>6s} {'first half':>11s} {'last half':>10s}")
    for d in args.run_dirs:
        eps = load_episodes(d)[args.eval_set]
        ok = [float(e["rollouts"][0]["success"]) for e in eps]  # execution order
        half = len(ok) // 2
        sr, first, last = 100 * np.mean(ok), 100 * np.mean(ok[:half]), 100 * np.mean(ok[half:])
        srs.append(sr)
        gains.append(last - first)
        print(f"{d:50s} {sr:6.1f} {first:11.1f} {last:10.1f}")
    ddof = 1 if len(srs) > 1 else 0
    print(f"\nSR mean +- std over {len(srs)} orders: {np.mean(srs):.1f} +- {np.std(srs, ddof=ddof):.1f}")
    print(f"last half - first half: mean {np.mean(gains):.1f}, min {np.min(gains):.1f}, max {np.max(gains):.1f}")


if __name__ == "__main__":
    main()
