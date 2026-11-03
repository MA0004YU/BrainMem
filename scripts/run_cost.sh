#!/usr/bin/env bash
# Table XV and Section IV-D: tokens, latency, retry overhead, memory growth.
# Every run logs these quantities; this script runs the two configurations on
# the EB-ALFRED base and long-horizon sets and prints the summary.
source "$(dirname "$0")/common.sh"
MODEL=${MODEL:-gpt-4o}
brainmem_run eb-alf "$MODEL" base cost_baseline --eval_sets base,long_horizon
brainmem_run eb-alf "$MODEL" full cost_brainmem --eval_sets base,long_horizon
conda activate embench
python "$(dirname "$0")/../tools/summarize.py" running/eb_alfred/${MODEL}_cost_baseline running/eb_alfred/${MODEL}_cost_brainmem
