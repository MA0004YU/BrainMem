# BrainMem: Brain-Inspired Evolving Memory for Embodied Agent Task Planning

This repository contains the implementation of **BrainMem**, a training-free hierarchical memory system
(working, episodic and semantic memory) that plugs into multi-modal LLM planners, together with the
configurations and scripts used for the experiments of the paper on
[EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench) (EB-ALFRED, EB-Habitat, EB-Navigation,
EB-Manipulation).

BrainMem does not modify EmbodiedBench. The benchmark environments, planners, prompts, in-context examples
and success metrics are EmbodiedBench's own; BrainMem replaces only the per-episode loop of each
EmbodiedBench evaluator and appends the retrieved memory context to the planner prompt.

```
BrainMem/
├── brainmem/
│   ├── config.py               hyper-parameters (Sec. IV-A) and ablation presets (Tables VIII-XIII)
│   ├── memory/
│   │   ├── working_memory.py   M_work: sliding window of m_t = (Action, Feedback, State, Location, t)   Sec. III-B-1
│   │   ├── trajectory_kg.py    M_epi Trajectory KG: state nodes, transition edges, merge rules        Sec. III-B-2
│   │   ├── spatial_kg.py       M_epi Spatial KG: (room contains obj), (obj_i r obj_j), support counts Sec. III-B-2
│   │   └── semantic_memory.py  M_sem: experience records, guidelines, utility scoring, pruning       Sec. III-B-3, III-D
│   ├── brainmem.py             step-/episode-level updates and hierarchical retrieval                 Sec. III-C, III-D
│   ├── experience_agent.py     LLM consolidation (one call per episode)                               Sec. III-B-3
│   ├── retry.py                guided-retry specification, failure-reason extraction, cost bound      Sec. III-B-3
│   ├── planner.py              appends the memory context to the EmbodiedBench planner prompt
│   ├── llm.py                  text access to the backbone used by the experience agent
│   ├── profiling.py            wall-clock and token accounting                                         Sec. IV-D
│   ├── adapters/               episode loop + one adapter per benchmark (alfred, habitat, navigation, manipulation)
│   └── run.py                  command-line entry point
├── configs/brainmem.yaml       hyper-parameters used for every reported run
├── scripts/                    one script per group of tables
├── tools/                      summarize.py (all metrics), order_robustness.py (Table XIV)
└── tests/                      offline tests (no simulator, no API)
```

## 1. Installation

1. Install EmbodiedBench by following its README (conda environments `embench`, `embench_nav`,
   `embench_man`, simulators and datasets). BrainMem uses EmbodiedBench as an installed package.
2. Install BrainMem into each of the three environments:

   ```bash
   git clone <this repository> BrainMem
   for env in embench embench_nav embench_man; do
     conda activate $env && pip install -e ./BrainMem
   done
   ```

3. Export the API keys exactly as for EmbodiedBench (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
   `GEMINI_API_KEY`; `remote_url` for open-source models served with LMDeploy).

All commands below are executed from the root of the EmbodiedBench checkout, because EmbodiedBench resolves
simulator assets relative to it. EB-ALFRED requires the X server described in the EmbodiedBench README.

## 2. Running BrainMem

```bash
conda activate embench
python -m brainmem.run --env eb-alf --model_name gpt-4o --preset full --eval_sets base
```

* `--env` is one of `eb-alf`, `eb-hab`, `eb-nav`, `eb-man`; use the conda environment EmbodiedBench
  prescribes for it.
* `--preset base` runs the unmodified EmbodiedBench planner (single attempt, no memory).
  `--preset full` is BrainMem. The ablation presets are listed in Section 5.
* All EmbodiedBench options (`--model_type`, `--n_shots`, `--resolution`, `--selected_indexes`, ...) are
  accepted with their EmbodiedBench meaning; unspecified options take the values of
  `embodiedbench/configs/<env>.yaml`.
* BrainMem hyper-parameters are read from `configs/brainmem.yaml` and can be overridden with
  `--set key=value`, e.g. `--set window_size=3`.
* The planner is decoded with temperature 0 and a maximum completion length of 2048 tokens. The image
  resolution (500×500) and the step budgets (30 / 20 / 15) are EmbodiedBench's defaults.

Memory starts empty for every run and is persisted as plain JSON in
`running/brainmem/<env>/<model>_<exp_name>/memory/`. Within one run the memory accumulates over the eval
sets in the order given by `--eval_sets`; separate runs (and therefore separate benchmarks) never share
memory. `--keep_memory 1` continues from an existing memory directory.

## 3. Outputs

EmbodiedBench's log directory of the run (e.g. `running/eb_alfred/gpt-4o_brainmem/base/`) contains

* `results/episode_<i>_final_res.json` and `results/summary.json` — EmbodiedBench-format metrics of the
  **first attempt** τ(0): `task_success`, `task_progress` (subgoal success), `num_steps` (environment
  steps), `planner_steps`, followed by BrainMem fields: `num_retries`, `final_success`, token counts per
  phase and wall-clock components;
* `brainmem/episode_<i>.json` — every rollout of the episode (actions, feedback, termination cause,
  failure reasons), the consolidation result, latencies, token usage and memory statistics;
* `retry_<r>/` — images and EmbodiedBench step logs of the guided-retry rollouts.

`python tools/summarize.py <log dir> [...]` aggregates the metrics of Tables I–III, VI, VII, XV and
Section IV-D per eval set: SR, SSR, planner / environment / total steps, retries per episode with the bound
R(1−SR₀), share of retry rollouts and of retry time, tokens of τ(0) plus consolidation and of retries,
per-decision latency, retrieval and update latency (incl. their first and last decile over the run),
consolidation time, total wall-clock time and its decomposition (planner inference, environment
simulation, consolidation, retrieval and updates), and storage size and graph sizes after the last episode.

## 4. Method ↔ code

| Paper | Code |
|---|---|
| m_t = (Action_t, Feedback_t, State_t, Location_t, t), window W | `WorkingMemory.append`, `WorkingMemoryEntry`, `window_size` |
| WM serialization: history, cumulative success rate, robot state, consistency warnings | `WorkingMemory.render`; benchmark-specific warnings in `adapters/*_consistency` |
| s_i = (state_id, room_id, summary); e_{i→j} = (prev, next, action, success, …) | `TrajectoryKG.states`, `TrajectoryKG.edges` |
| s_0 from initial scene and instruction; per step create/reuse s_t, add e_{t−1→t} | `TrajectoryKG.start_rollout`, `TrajectoryKG.add_step` |
| state deduplication / transition aggregation / failure tagging | `_match_or_create_state` (Jaccard ≥ `state_merge_threshold`), counters and `reasoning_examples` in `add_step`, `failed` tag |
| (room contains object_i), (object_i r object_j), support count | `SpatialKG.observe`, `SpatialKG.consolidate` |
| M_epi capacity: bounded by merging, ranked by success / support count | `TrajectoryKG.recall_sequences`, `failed_transitions`, `SpatialKG.retrieve` |
| experience record (instruction, room, status, action sequence, failure reasons, success pattern) | `BrainMem.consolidate` → `SemanticMemory.add_record` |
| guided retry: reset, clear WM, keep M_epi/M_sem, failure reasons of τ(0..r−1), stop on success, SR on τ(0) | `BrainMemEvaluatorMixin.evaluate`, `retry.py` |
| at most R+1 rollouts, expected retries ≤ R(1−SR₀) | `retry.expected_retry_bound`; measured by `tools/summarize.py` |
| corrective guideline from the contrast between the successful retry and its predecessors | `ExperienceAgent.build_prompt` / `consolidate` |
| confidence(g) = N_success/N_total, usage(g) = min(N_success/10, 1), utility = α·conf + β·usage | `semantic_memory.confidence / usage / utility` |
| counts updated after each episode in which g is applied | `SemanticMemory.record_application` (guidelines injected into τ(0), outcome of τ(0)) |
| at most M guidelines, pruning above 1.5 M, protection for conf ≥ 0.8 and N_success ≥ 5 | `SemanticMemory.prune` |
| hierarchical retrieval for Q = {room, task_type, object} | `BrainMem.context` |
| step-level updates (WM → Trajectory KG → Spatial KG) | `BrainMem.after_step` |
| episode-level updates 1–4 | `BrainMem.end_rollout`, `BrainMem.consolidate` |

The task query uses the benchmark's own task metadata: EB-ALFRED task type and target object come from the
ALFRED task identifier (e.g. `pick_clean_then_place_in_recep-Ladle-…`), EB-Navigation from the target object
of the episode, EB-Manipulation from the task variation; EB-Habitat task types are derived from the number
of objects named in the instruction and whether a receptacle has to be opened. The room key is the
simulator scene (AI2-THOR floor plan, Habitat scene, EB-Manipulation task scene).

Hyper-parameters (all runs, all benchmarks): W = 5, k = 2, α = 0.7, β = 0.3, M = 20 with pruning above 30
guidelines, protection at confidence ≥ 0.8 and N_success ≥ 5, R = 3. The experience agent is the backbone
planner of the run. The remaining implementation constants (similarity thresholds, numbers of recalled
items) are listed with their meaning in `configs/brainmem.yaml` and `brainmem/config.py`.

## 5. Reproducing the experiments

| Paper | Script | Notes |
|---|---|---|
| Tables I, II, III, VI, VII | `scripts/run_main.sh` | GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro; `base` and `full` on all subsets |
| Tables IV, V | `scripts/run_backbones.sh` | Qwen2.5-VL-72B-Instruct (LMDeploy), GPT-4o, Gemini-3.5-Flash; Base and Long |
| Table VIII | `scripts/run_ablation.sh` | presets `base`, `wm`, `wm_fb`, `wm_em`, `wm_em_fb`, `wm_em_sm`, `full` |
| Table IX | `scripts/run_ablation.sh` | `full_no_feedback` (= `wm_em_sm`), `full_no_success_state`, `full_no_agent_state`, `full_no_consistency_check` |
| Table X | `scripts/run_ablation.sh` | `--set window_size=W`, W ∈ {1, 3, 5, 8, 10} |
| Table XI | `scripts/run_ablation.sh` | `full_no_spatial_kg`, `full_no_trajectory_kg` |
| Table XII | `scripts/run_ablation.sh` | `--set top_k=k`, k ∈ {1, 2, 3, 5} |
| Table XIII | `scripts/run_ablation.sh` | `full_no_success_patterns`, `full_no_failure_reasons`, `full_no_learning_insights` |
| Table XIV | `scripts/run_order_robustness.sh` | `--episode_order_seed s`, s = 1…5; SR on first vs. last half of the order |
| Table XV, Sec. IV-D | `scripts/run_cost.sh` | tokens, latency, retry overhead, storage; logged by every run |

Ablation switches:

* **Working / Episodic / Semantic memory** enable M_work, M_epi (both KGs) and M_sem (records, guidelines,
  experience agent).
* **Feedback** enables the feedback-based evolving of Section III-D: execution feedback (success/failure and
  the simulator message) is recorded in the working-memory tuples and in the Trajectory KG (success counts,
  failure tagging, outcome-ranked recall), guideline counts are updated from episode outcomes, and failed
  first attempts trigger guided retry. Without feedback the memory stores actions, states and observations
  only and every episode has a single rollout. The "success state" of Table IX is the cumulative success
  rate, which is computed from feedback.
* **Learning insights** (Table XIII) covers the learning insights of the experience records and the
  guidelines distilled from them; **successful patterns** and **failure reasons** are the corresponding
  fields of the retrieved experience records.

## 6. Tests

```bash
python -m unittest discover -s tests -v
```

The tests run without simulators or API access. They check the definitions and the numerical example of
Section III-D (utility crossing at α = 7/9 ≈ 0.78, retry bound 3 × (1 − 0.82) = 0.54), pruning with
protection, the three merge rules of the Trajectory KG, per-episode support counting in the Spatial KG,
working-memory contents, the preset definitions of Table VIII, and the complete episode loop
(first-attempt reporting, guided retry, a single consolidation call per episode, guideline reuse).

## 7. Experimental platform

All experiments run on a workstation with 1× NVIDIA RTX A6000 48 GB, dual AMD EPYC 7513 32-core CPUs and
256 GB of system memory for simulation rendering. Open-source planners are served on a separate node with
4× NVIDIA H100 80 GB via LMDeploy 0.7.0 under PyTorch 2.4.0 and CUDA 12.1; proprietary planners are accessed
through their official APIs at frozen checkpoints. The simulation stack and the EmbodiedBench evaluation
harness are those listed in Section IV-A of the paper, installed with EmbodiedBench's installation
procedure; the official splits and episode orderings are used without modification. Every run writes the
software versions, GPU and git commit to `running/brainmem/<env>/<model>_<exp_name>/run_config.json`.

## 8. Demos

[`docs/brainmem_demo.html`](docs/brainmem_demo.html) is a self-contained interactive replay of one real-robot
recording on a Franka FR3 in which three pick-and-place tasks are run back to back (iron box → tray A,
stapler → tray C, plug → tray B). Over the 706 s recording it synchronises the fixed camera view, the vertical
force trace, the current motion intent and the contents of working, scene, episodic and semantic memory, and
marks for each task when memory is read (known tray regions, retrieval of similar episodes), when the result is
written back and when consolidation finishes. Download the file and open it in a browser.

[`docs/brainmem_demo_simulator.mp4`](docs/brainmem_demo_simulator.mp4) is a 95 s simulator demo of the task
"put the two newspapers into the drawer". It shows the agent's actions step by step and, after BrainMem is
introduced, the working, episodic and long-term memory used for the task and the experience summarised from it.

## License

BrainMem is released under the MIT License (see `LICENSE`). EmbodiedBench, the simulators and the datasets
are used under their own licenses.
