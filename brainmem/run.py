"""Command-line entry point.

Run from the root of the EmbodiedBench checkout (EmbodiedBench resolves
simulator assets relative to it), inside the conda environment EmbodiedBench
prescribes for the benchmark (embench / embench_nav / embench_man):

    python -m brainmem.run --env eb-alf --model_name gpt-4o --preset full --eval_sets base

The EmbodiedBench benchmark settings come from
``embodiedbench/configs/<env>.yaml`` and can be overridden with the usual
EmbodiedBench options; BrainMem hyper-parameters come from
``configs/brainmem.yaml`` of this repository, the selected ``--preset``, and
``--set key=value`` overrides.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BM_CONFIG = os.path.join(REPO_ROOT, "configs", "brainmem.yaml")

EVALUATORS = {
    "eb-alf": ("brainmem.adapters.alfred", "BrainMemAlfredEvaluator"),
    "eb-hab": ("brainmem.adapters.habitat", "BrainMemHabitatEvaluator"),
    "eb-nav": ("brainmem.adapters.navigation", "BrainMemNavigationEvaluator"),
    "eb-man": ("brainmem.adapters.manipulation", "BrainMemManipulationEvaluator"),
}

# Decoding settings of all reported runs (Section IV-A).
PLANNER_TEMPERATURE = 0
PLANNER_MAX_COMPLETION_TOKENS = 2048


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        low = text.lower()
        if low in ("true", "false"):
            return low == "true"
        return text


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BrainMem on EmbodiedBench")
    p.add_argument("--env", required=True, choices=sorted(EVALUATORS))
    p.add_argument("--preset", default="full", help="see brainmem/config.py:PRESETS")
    p.add_argument("--bm_config", default=DEFAULT_BM_CONFIG, help="BrainMem hyper-parameter YAML")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a BrainMem hyper-parameter, e.g. --set window_size=3")
    p.add_argument("--keep_memory", type=int, default=0,
                   help="1 = continue from an existing memory directory instead of starting empty")
    p.add_argument("--episode_order_seed", type=int, default=None,
                   help="EB-ALFRED: evaluate the episodes in a random order drawn with this seed (Table XIV)")
    # EmbodiedBench options (same names as embodiedbench/main.py)
    p.add_argument("--model_name", type=str)
    p.add_argument("--model_type", type=str)
    p.add_argument("--exp_name", type=str)
    p.add_argument("--eval_sets", type=lambda s: [x for x in s.split(",") if x])
    p.add_argument("--selected_indexes", type=lambda s: [int(x) for x in s.split(",") if x])
    p.add_argument("--down_sample_ratio", type=float)
    p.add_argument("--n_shots", type=int)
    p.add_argument("--resolution", type=int)
    p.add_argument("--language_only", type=int)
    p.add_argument("--chat_history", type=int)
    p.add_argument("--multistep", type=int)
    p.add_argument("--multiview", type=int)
    p.add_argument("--detection_box", type=int)
    p.add_argument("--visual_icl", type=int)
    p.add_argument("--env_feedback", type=int)
    p.add_argument("--tp", type=int)
    return p.parse_args(argv)


def load_benchmark_config(env: str, args: argparse.Namespace) -> Dict[str, Any]:
    import embodiedbench
    # embodiedbench is a namespace package (no __init__.py), so use __path__.
    path = os.path.join(list(embodiedbench.__path__)[0], "configs", f"{env}.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("model_name", "model_type", "exp_name", "eval_sets", "selected_indexes", "down_sample_ratio",
                "n_shots", "resolution", "language_only", "chat_history", "multistep", "multiview",
                "detection_box", "visual_icl", "env_feedback", "tp"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.exp_name is None:
        cfg["exp_name"] = f"brainmem_{args.preset}"
    cfg.setdefault("selected_indexes", [])
    if args.episode_order_seed is not None:
        cfg["episode_order_seed"] = args.episode_order_seed
    return cfg


def system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": sys.version, "platform": platform.platform(),
                            "processor": platform.processor()}
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                      "--format=csv,noheader"], capture_output=True, text=True,
                                     timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["gpu"] = ""
    for module in ("ai2thor", "habitat_sim", "habitat", "torch", "openai", "anthropic", "lmdeploy"):
        try:
            mod = __import__(module)
            info[f"version_{module}"] = getattr(mod, "__version__", "")
        except Exception:
            info[f"version_{module}"] = "not installed"
    try:
        info["brainmem_git_commit"] = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                                                     capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["brainmem_git_commit"] = ""
    return info


def main(argv=None) -> None:
    args = parse_args(argv)
    from brainmem.config import build_config

    with open(args.bm_config) as f:
        bm_base = yaml.safe_load(f) or {}
    overrides = {}
    for item in args.overrides:
        key, _, value = item.partition("=")
        overrides[key.strip()] = _parse_value(value.strip())
    bm_cfg = build_config(args.preset, overrides, bm_base)
    config = load_benchmark_config(args.env, args)

    import embodiedbench.planner.remote_model as remote_model
    remote_model.temperature = PLANNER_TEMPERATURE
    remote_model.max_completion_tokens = PLANNER_MAX_COMPLETION_TOKENS

    module_name, class_name = EVALUATORS[args.env]
    module = __import__(module_name, fromlist=[class_name])
    evaluator_cls = getattr(module, class_name)

    model_short = config["model_name"].split("/")[-1]
    run_dir = os.path.join("running", "brainmem", args.env, f"{model_short}_{config['exp_name']}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump({"preset": args.preset, "benchmark_config": config, "brainmem": bm_cfg.to_dict(),
                   "planner_temperature": PLANNER_TEMPERATURE,
                   "planner_max_completion_tokens": PLANNER_MAX_COMPLETION_TOKENS,
                   "argv": sys.argv, "system": system_info()}, f, indent=2, default=str)

    evaluator = evaluator_cls(config)
    evaluator.setup_brainmem(bm_cfg, run_dir, fresh_memory=not args.keep_memory)
    evaluator.check_config_valid()
    evaluator.evaluate_main()


if __name__ == "__main__":
    main()
