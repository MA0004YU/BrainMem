#!/usr/bin/env bash
# Tables I-III, VI-VII: the three backbones with and without BrainMem on all
# subsets of the four benchmarks. "base" is the plain EmbodiedBench planner.
source "$(dirname "$0")/common.sh"
MODELS=${MODELS:-"gpt-4o claude-3-5-sonnet-20241022 gemini-1.5-pro"}
ENVS=${ENVS:-"eb-alf eb-hab eb-nav eb-man"}
for env in $ENVS; do
  for model in $MODELS; do
    brainmem_run "$env" "$model" base baseline --eval_sets "$(subsets_for "$env")"
    brainmem_run "$env" "$model" full brainmem --eval_sets "$(subsets_for "$env")"
  done
done
