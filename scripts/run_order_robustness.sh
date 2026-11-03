#!/usr/bin/env bash
# Table XIV: five random episode orders of the EB-ALFRED base set (full model, GPT-4o).
source "$(dirname "$0")/common.sh"
MODEL=${MODEL:-gpt-4o}
for seed in 1 2 3 4 5; do
  brainmem_run eb-alf "$MODEL" full "order_seed${seed}" --eval_sets base --episode_order_seed "$seed"
done
conda activate embench
python "$(dirname "$0")/../tools/order_robustness.py" running/eb_alfred/${MODEL}_order_seed{1,2,3,4,5}
