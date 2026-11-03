"""BrainMem configuration.

Every field maps to a quantity defined in the paper. Defaults are the values
used for all reported experiments (Section IV-A, "Implementation Details"):

    W = 5                 working-memory window
    k = 2                 retrieval depth for semantic memory
    alpha = 0.7, beta = 0.3   utility weights, alpha + beta = 1
    M = 20                guideline capacity, pruning triggered at 1.5 M entries
    protection            confidence >= 0.8 and N_success >= 5
    R = 3                 guided-retry budget (retries after the first attempt)

The ``use_*`` switches select the memory components and are the switches used
by the ablation presets of Tables VIII, IX, XI and XIII (see ``PRESETS``).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional


@dataclass
class BrainMemConfig:
    # ---- component switches (Table VIII) ---------------------------------
    enabled: bool = True              # False reproduces the plain EmbodiedBench planner
    use_working_memory: bool = True   # M_work
    use_episodic_memory: bool = True  # M_epi (Trajectory KG + Spatial KG)
    use_semantic_memory: bool = True  # M_sem (experience records + guidelines)
    use_feedback: bool = True         # feedback-based evolving incl. guided retry

    # ---- working memory contents (Table IX) / window (Table X) -------------
    window_size: int = 5              # W
    wm_success_state: bool = True     # aggregate statistics (cumulative success rate)
    wm_agent_state: bool = True       # current robot state (e.g. what the agent holds)
    wm_consistency_check: bool = True # consistency warnings for invalid action sequences

    # ---- episodic memory (Table XI) ---------------------------------------
    use_trajectory_kg: bool = True
    use_spatial_kg: bool = True
    # State deduplication: two states in the same room are merged when the
    # Jaccard similarity of their summary token sets reaches this value.
    state_merge_threshold: float = 0.8
    # Two past tasks are "highly similar" (Trajectory-KG recall, Sec. III-C)
    # when they share the task type and either the target object or an
    # instruction-token Jaccard similarity of at least this value.
    task_similarity_threshold: float = 0.5
    max_reasoning_examples: int = 3   # reasoning examples attached per aggregated transition
    traj_recall_sequences: int = 2    # recalled action sequences per planning call
    traj_recall_max_actions: int = 15 # actions shown per recalled sequence
    failed_transitions_shown: int = 3 # failure-tagged transitions shown per planning call
    spatial_relations_shown: int = 12 # top relations of the current room shown per call
    spatial_stable_support: int = 2   # relations confirmed by >= this many episodes are "stable"

    # ---- semantic memory (Tables XII, XIII) ---------------------------------
    top_k: int = 2                    # k
    sm_success_patterns: bool = True
    sm_failure_reasons: bool = True
    sm_learning_insights: bool = True # learning insights and the guidelines distilled from them
    alpha: float = 0.7
    beta: float = 0.3
    usage_saturation: int = 10        # usage(g) = min(N_success / 10, 1)
    max_guidelines: int = 20          # M
    prune_factor: float = 1.5         # pruning is triggered above prune_factor * M
    protect_confidence: float = 0.8
    protect_min_success: int = 5
    guideline_merge_threshold: float = 0.6  # token Jaccard above which two guidelines are merged
    distill_context_records: int = 3  # compatible experience records shown to the experience agent

    # ---- guided retry (Section III-B) --------------------------------------
    max_retries: int = 3              # R

    # ---- experience agent ---------------------------------------------------
    # None -> the backbone planner model of the run (paper setting).
    experience_model: Optional[str] = None
    experience_max_tokens: int = 2048
    experience_temperature: float = 0.0

    # ---- persistence --------------------------------------------------------
    memory_dir: str = ""              # filled in by the runner when empty

    def validate(self) -> None:
        if abs(self.alpha + self.beta - 1.0) > 1e-9:
            raise ValueError(f"alpha + beta must equal 1 (got {self.alpha} + {self.beta})")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.prune_factor < 1.0:
            raise ValueError("prune_factor must be >= 1")

    @property
    def prune_trigger(self) -> int:
        """Soft upper limit 1.5 M at which pruning runs."""
        return int(self.prune_factor * self.max_guidelines)

    @property
    def retry_enabled(self) -> bool:
        return self.enabled and self.use_feedback and self.max_retries > 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrainMemConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise KeyError(f"Unknown BrainMem config keys: {sorted(unknown)}")
        return cls(**data)


# Presets used in the paper. Each preset is a set of overrides on top of the
# default (full) configuration.
PRESETS: Dict[str, Dict[str, Any]] = {
    # Table VIII -- memory modules and feedback
    "base": {"enabled": False},
    "wm": {"use_episodic_memory": False, "use_semantic_memory": False, "use_feedback": False},
    "wm_fb": {"use_episodic_memory": False, "use_semantic_memory": False},
    "wm_em": {"use_semantic_memory": False, "use_feedback": False},
    "wm_em_fb": {"use_semantic_memory": False},
    "wm_em_sm": {"use_feedback": False},
    "full": {},
    # Table IX -- contents of working memory
    "full_no_feedback": {"use_feedback": False},
    "full_no_success_state": {"wm_success_state": False},
    "full_no_agent_state": {"wm_agent_state": False},
    "full_no_consistency_check": {"wm_consistency_check": False},
    # Table XI -- knowledge graphs in episodic memory
    "full_no_spatial_kg": {"use_spatial_kg": False},
    "full_no_trajectory_kg": {"use_trajectory_kg": False},
    # Table XIII -- components of semantic memory
    "full_no_success_patterns": {"sm_success_patterns": False},
    "full_no_failure_reasons": {"sm_failure_reasons": False},
    "full_no_learning_insights": {"sm_learning_insights": False},
}


def build_config(preset: str = "full", overrides: Optional[Dict[str, Any]] = None,
                 base: Optional[Dict[str, Any]] = None) -> BrainMemConfig:
    """Compose ``base`` (e.g. configs/brainmem.yaml) <- preset <- overrides."""
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset '{preset}'. Available: {sorted(PRESETS)}")
    data: Dict[str, Any] = copy.deepcopy(base or {})
    data.update(PRESETS[preset])
    data.update(overrides or {})
    cfg = BrainMemConfig.from_dict(data)
    cfg.validate()
    return cfg
