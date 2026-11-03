#!/usr/bin/env bash
# Tables IV-V: BrainMem with further backbones on the Base and Long subsets.
# Open-source planners are served with LMDeploy behind an OpenAI-compatible
# endpoint (export remote_url=http://<host>:<port>/v1), as in EmbodiedBench.
source "$(dirname "$0")/common.sh"
# Table IV: Qwen2.5-VL-72B-Instruct on EB-ALFRED and EB-Habitat
for env in eb-alf eb-hab; do
  brainmem_run "$env" Qwen/Qwen2.5-VL-72B-Instruct full brainmem --eval_sets base,long_horizon
done
# Table V: GPT-4o and Gemini-3.5-Flash on EB-ALFRED
: "${GEMINI_FLASH_MODEL:?set GEMINI_FLASH_MODEL to the provider model id of Gemini-3.5-Flash}"
for model in gpt-4o "$GEMINI_FLASH_MODEL"; do
  brainmem_run eb-alf "$model" full brainmem --eval_sets base,long_horizon
done
