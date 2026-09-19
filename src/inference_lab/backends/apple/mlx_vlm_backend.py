"""Text-only Qwen3.5 inference using MLX-VLM's model implementation."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .mlx_backend import MLXBackend


class MLXVLMBackend(MLXBackend):
    """Compare MLX-VLM and MLX-LM on the same weights and timing harness.

    These are different Python model implementations sharing the MLX/Metal
    compute runtime. Images and video are outside this text-only workload.
    The unused vision module is discarded before weights are materialized.
    """

    def load(self) -> MLXVLMBackend:
        if self._model is not None:
            return self
        model_path = Path(self.model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {self.model_path}")
        try:
            import mlx.core as mx
            from mlx_vlm.utils import load_model
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MLX-VLM is unavailable. Run the project's environment setup script."
            ) from exc
        if not mx.metal.is_available():
            raise RuntimeError("This backend requires Apple Silicon with Metal available")
        mx.set_default_device(mx.gpu)
        multimodal_model = load_model(
            model_path,
            lazy=True,
            trust_remote_code=False,
            local_files_only=True,
        )
        language_model = multimodal_model.language_model
        language_model.eval()
        # Hold only the component used by this workload, so an idle vision tower
        # does not distort the measured memory footprint relative to MLX-LM.
        del multimodal_model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=False, local_files_only=True
        )
        mx.eval(language_model.parameters())
        mx.synchronize()
        mx.clear_cache()
        self._mx = mx
        self._model = language_model
        self._tokenizer = tokenizer
        self._make_prompt_cache = lambda model: model.make_cache()
        self._model_config = json.loads((model_path / "config.json").read_text())
        return self

    def _next_token(self, input_tokens: Any, cache: list[Any]) -> Any:
        output = self._model(input_tokens[None], cache=cache)
        self._quantize_cache(cache)
        return self._mx.argmax(output.logits[:, -1, :], axis=-1)

    def measure(
        self, prompt_tokens: list[int], max_new_tokens: int = 128
    ) -> dict[str, Any]:
        # MLX-VLM keeps rotary position bookkeeping on the language module as
        # well as the cache. Reset both between independent dataset requests.
        if self._model is not None:
            for attribute in ("_position_ids", "_rope_deltas"):
                if hasattr(self._model, attribute):
                    setattr(self._model, attribute, None)
        return super().measure(prompt_tokens, max_new_tokens)

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata["framework"] = "mlx-vlm"
        metadata["versions"].pop("mlx-lm", None)
        for package in ("mlx-vlm", "transformers"):
            try:
                metadata["versions"][package] = version(package)
            except PackageNotFoundError:
                metadata["versions"][package] = None
        metadata.update(
            {
                "compute_runtime": "mlx / Metal (shared with mlx-lm)",
                "model_implementation": (
                    f"{type(self._model).__module__}.{type(self._model).__name__}"
                    if self._model is not None
                    else None
                ),
                "model_loader": "mlx_vlm.utils.load_model",
                "vision_tower_retained": False,
                "measurement_harness": "shared phase-separated MLX harness",
            }
        )
        return metadata
