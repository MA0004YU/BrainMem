#!/usr/bin/env bash
# Shared settings for the experiment scripts. Run the scripts from the root of
# the EmbodiedBench checkout, e.g.
#   cd /path/to/EmbodiedBench && bash /path/to/BrainMem/scripts/run_main.sh
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"

# conda environment prescribed by EmbodiedBench for each benchmark
conda_env_for() {
  case "$1" in
    eb-alf|eb-hab) echo embench ;;
    eb-nav) echo embench_nav ;;
    eb-man) echo embench_man ;;
  esac
}

# all eval subsets used in the paper
subsets_for() {
  case "$1" in
    eb-alf|eb-hab) echo base,common_sense,complex_instruction,visual_appearance,spatial,long_horizon ;;
    eb-nav) echo base,common_sense,complex_instruction,visual_appearance,long_horizon ;;
    eb-man) echo base,common_sense,complex,spatial,visual ;;
  esac
}

# brainmem_run <env> <model> <preset> <exp_name> [extra args...]
brainmem_run() {
  local env="$1" model="$2" preset="$3" exp="$4"; shift 4
  conda activate "$(conda_env_for "$env")"
  python -m brainmem.run --env "$env" --model_name "$model" --preset "$preset" --exp_name "$exp" "$@"
}
