#!/usr/bin/env bash
# Tables VIII-XIII: ablations with GPT-4o on the EB-ALFRED base set.
source "$(dirname "$0")/common.sh"
MODEL=${MODEL:-gpt-4o}
# Table VIII: memory modules and feedback
for p in base wm wm_fb wm_em wm_em_fb wm_em_sm full; do
  brainmem_run eb-alf "$MODEL" "$p" "abl_${p}" --eval_sets base
done
# Table IX: contents of working memory (the full model and "w/o feedback" are rows of Table VIII)
for p in full_no_success_state full_no_agent_state full_no_consistency_check; do
  brainmem_run eb-alf "$MODEL" "$p" "abl_${p}" --eval_sets base
done
# Table X: working-memory window size (W = 5 is the full model)
for w in 1 3 8 10; do
  brainmem_run eb-alf "$MODEL" full "abl_window${w}" --eval_sets base --set window_size=$w
done
# Table XI: knowledge graphs in episodic memory
for p in full_no_spatial_kg full_no_trajectory_kg; do
  brainmem_run eb-alf "$MODEL" "$p" "abl_${p}" --eval_sets base
done
# Table XII: retrieval depth k of semantic memory (k = 2 is the full model)
for k in 1 3 5; do
  brainmem_run eb-alf "$MODEL" full "abl_topk${k}" --eval_sets base --set top_k=$k
done
# Table XIII: components of semantic memory
for p in full_no_success_patterns full_no_failure_reasons full_no_learning_insights; do
  brainmem_run eb-alf "$MODEL" "$p" "abl_${p}" --eval_sets base
done
