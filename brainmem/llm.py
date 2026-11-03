"""Text-only access to the backbone model used by the experience agent.

Experience summarization and guideline distillation are performed by the same
backbone planner used for the corresponding run, so BrainMem introduces no
additional model dependency. ``BackboneLLM.from_planner`` therefore reuses the
SDK client (or the LMDeploy pipeline) that EmbodiedBench's planner already
holds. Passing ``experience_model`` builds a client with the same provider
routing as ``embodiedbench.planner.remote_model.RemoteModel``.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

Usage = Dict[str, int]


class BackboneLLM:
    def __init__(self, model_name: str, model_type: str, client: Any,
                 max_tokens: int = 2048, temperature: float = 0.0, max_attempts: int = 3):
        self.model_name = model_name
        self.model_type = model_type
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_attempts = max_attempts

    @classmethod
    def from_planner(cls, planner: Any, experience_model: Optional[str] = None,
                     max_tokens: int = 2048, temperature: float = 0.0) -> "BackboneLLM":
        remote = planner.model  # embodiedbench RemoteModel
        if getattr(planner, "model_type", "remote") == "custom":
            raise ValueError("BrainMem supports planners with model_type 'remote' or 'local'")
        if experience_model and experience_model != planner.model_name:
            from embodiedbench.planner.remote_model import RemoteModel
            remote = RemoteModel(experience_model, "remote", language_only=True)
            return cls(experience_model, "remote", remote.model, max_tokens, temperature)
        return cls(planner.model_name, getattr(remote, "model_type", "remote"), remote.model,
                   max_tokens, temperature)

    # ------------------------------------------------------------------
    def complete(self, system: str, user: str) -> Tuple[str, Usage]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                return self._complete_once(system, user)
            except Exception as e:  # network / rate-limit errors of the provider SDKs
                last_error = e
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"experience agent call failed after {self.max_attempts} attempts: {last_error}")

    def _complete_once(self, system: str, user: str) -> Tuple[str, Usage]:
        if self.model_type == "local":
            from lmdeploy import GenerationConfig
            response = self.client(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                gen_config=GenerationConfig(temperature=self.temperature, max_new_tokens=self.max_tokens))
            return response.text, {"input_tokens": int(response.input_token_len),
                                   "output_tokens": int(response.generate_token_len)}
        if "claude" in self.model_name:
            response = self.client.messages.create(
                model=self.model_name, max_tokens=self.max_tokens, temperature=self.temperature,
                system=system, messages=[{"role": "user", "content": user}])
            text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
            return text, {"input_tokens": int(response.usage.input_tokens),
                          "output_tokens": int(response.usage.output_tokens)}
        kwargs: Dict[str, Any] = {"model": self.model_name,
                                  "messages": [{"role": "system", "content": system},
                                               {"role": "user", "content": user}]}
        if self.model_name.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature
        response = self.client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        return response.choices[0].message.content or "", {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0)}
