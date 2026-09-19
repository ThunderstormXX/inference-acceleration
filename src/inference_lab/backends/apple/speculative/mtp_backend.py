"""Native Qwen3.5 MTP using stock MLX-VLM verification and full prompt features."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from ..mlx_vlm_backend import MLXVLMBackend


def _acceptance_metrics(accepted_lengths, draft_lengths, output_tokens):
    """Preserve verified acceptance separately from length-capped emission."""
    if len(accepted_lengths) != len(draft_lengths):
        raise RuntimeError("MTP acceptance and proposal histories have different lengths")
    remaining = output_tokens - 1
    rounds = []
    for accepted, drafted in zip(accepted_lengths, draft_lengths):
        if type(accepted) is not int or type(drafted) is not int or not 0 <= accepted <= drafted or drafted < 1:
            raise RuntimeError("Invalid MTP acceptance history")
        if remaining <= 0:
            raise RuntimeError("MTP recorded work after the requested output budget")
        emitted = min(accepted + 1, remaining)
        emitted_drafts = min(accepted, emitted)
        rounds.append({
            "drafted_tokens": drafted,
            "accepted_draft_tokens": accepted,
            "emitted_tokens": emitted,
            "emitted_draft_tokens": emitted_drafts,
            "emitted_target_tokens": emitted - emitted_drafts,
        })
        remaining -= emitted
    if remaining:
        raise RuntimeError("MTP acceptance history does not cover generated tokens")
    drafted = sum(item["drafted_tokens"] for item in rounds)
    accepted = sum(item["accepted_draft_tokens"] for item in rounds)
    return {
        "drafted_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "draft_acceptance_rate": accepted / drafted if drafted else None,
        "speculative_rounds": len(rounds),
        "mean_accepted_draft_tokens_per_round": accepted / len(rounds) if rounds else None,
        "emitted_draft_tokens": sum(item["emitted_draft_tokens"] for item in rounds),
        "emitted_target_tokens": 1 + sum(item["emitted_target_tokens"] for item in rounds),
        "speculative_round_details": rounds,
    }


class MTPBackend(MLXVLMBackend):
    """Use a separate native MTP head with the same target as MLXVLMBackend.

The installed generic chunked prefill retains complete prompt hidden states for
DFlash/EAGLE3 only. Here every chunk contributes its normalized hidden states,
and MTP receives the original full prompt and aligned features through the stock
round-loop API. Its verification, cache rollback and proposal math are unchanged.
"""

    def __init__(
        self,
        model_path: str,
        draft_path: str,
        prefill_step_size: int = 512,
        block_size: int = 3,
        wired_memory: bool = True,
    ) -> None:
        if type(block_size) is not int or not 2 <= block_size <= 5:
            raise ValueError("MTP block_size must be an integer between 2 and 5")
        if wired_memory is not True:
            raise ValueError("This paired MTP protocol requires wired_memory=True")
        super().__init__(model_path, prefill_step_size, kv_bits=None, wired_memory=True)
        self.draft_path = str(Path(draft_path).expanduser().resolve())
        self.block_size = block_size
        self._draft: Any = None
        self._run_rounds: Any = None
        self._generation_stream: Any = None
        self._draft_config: dict[str, Any] = {}
        self._draft_manifest: dict[str, Any] = {}
        self._draft_manifest_sha256: str | None = None
        self._runtime_sources: dict[str, dict[str, str]] = {}

    def load(self) -> MTPBackend:
        if self._model is not None and self._draft is not None:
            return self
        draft_path = Path(self.draft_path)
        self._draft_config = json.loads((draft_path / "config.json").read_text())
        if self._draft_config.get("model_type") != "qwen3_5_mtp":
            raise ValueError("Expected a native qwen3_5_mtp checkpoint")
        target_config = json.loads((Path(self.model_path) / "config.json").read_text())
        if target_config.get("model_type") != "qwen3_5":
            raise ValueError("The native MTP adapter supports Qwen3.5 targets only")
        target_text = target_config.get("text_config", target_config)
        draft_text = self._draft_config.get("text_config", {})
        for key in ("hidden_size", "num_hidden_layers", "vocab_size"):
            if draft_text.get(key) != target_text.get(key):
                raise ValueError(f"MTP draft/target {key} mismatch")
        manifest_bytes = (draft_path / "download-manifest.json").read_bytes()
        self._draft_manifest = json.loads(manifest_bytes)
        if not self._draft_manifest.get("repo_id") or not self._draft_manifest.get("resolved_revision"):
            raise ValueError("MTP manifest must identify its source repository and revision")
        self._draft_manifest_sha256 = sha256(manifest_bytes).hexdigest()

        from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
        from mlx_vlm.speculative import cache_state, common, mtp, utils
        from mlx_vlm.speculative.drafters.qwen3_5_mtp import qwen3_5_mtp
        from mlx_vlm.models.qwen3_5 import speculative_verifier

        super().load()
        draft, kind = load_drafter(
            self.draft_path, kind="mtp", lazy=True,
            trust_remote_code=False, local_files_only=True,
        )
        if kind != "mtp":
            raise RuntimeError(f"Unexpected resolved MTP draft kind: {kind}")
        validate_drafter_compatibility(self._model, draft, kind)
        draft.eval()
        self._mx.eval(draft.parameters())
        self._mx.synchronize()
        self._mx.clear_cache()
        self._draft = draft
        self._run_rounds = utils.run_speculative_rounds
        self._generation_stream = common.generation_stream
        for module in (cache_state, common, mtp, utils, qwen3_5_mtp, speculative_verifier):
            source = Path(module.__file__)
            self._runtime_sources[module.__name__] = {
                "path": str(source.resolve()), "sha256": sha256(source.read_bytes()).hexdigest(),
            }
        return self

    def _measure(self, prompt_tokens: list[int], max_new_tokens: int = 128) -> dict[str, Any]:
        # Inherited MLXVLMBackend.measure resets rotary bookkeeping and wraps
        # this operation in the same wired-memory policy as the paired baseline.
        if self._model is None or self._draft is None:
            raise RuntimeError("Call load() before measure()")
        if not prompt_tokens or any(type(token) is not int or token < 0 for token in prompt_tokens):
            raise ValueError("prompt_tokens must contain nonnegative integer token IDs")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self._draft.reset(self._model)
        try:
            return self._measure_mtp(prompt_tokens, max_new_tokens)
        finally:
            # Avoid retaining a previous request's drafter KV/seed during the
            # next prompt's prefill or carrying history across independent rows.
            self._draft.reset(self._model)

    def _measure_mtp(self, prompt_tokens: list[int], max_new_tokens: int) -> dict[str, Any]:
        mx = self._mx
        cache = self._make_prompt_cache(self._model)
        prompt = mx.array(prompt_tokens, dtype=mx.uint32)
        full_input_ids = prompt[None]
        mx.eval(prompt)
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        stream = None
        hidden_chunks = []
        generated = []
        try:
            started = perf_counter()
            with mx.stream(self._generation_stream):
                position = 0
                while position < len(prompt_tokens) - 1:
                    end = min(position + self.prefill_step_size, len(prompt_tokens) - 1)
                    output = self._model(
                        prompt[position:end][None], cache=cache,
                        return_hidden=True, return_shared_kv=True, skip_logits=True,
                    )
                    hidden_chunks.append(output.hidden_states[-1])
                    mx.eval(hidden_chunks[-1], [entry.state for entry in cache])
                    mx.clear_cache()
                    position = end
                output = self._model(
                    prompt[-1:][None], cache=cache,
                    return_hidden=True, return_shared_kv=True,
                )
                hidden_chunks.append(output.hidden_states[-1])
                output.hidden_states = [mx.concatenate(hidden_chunks, axis=1)]
                token = mx.argmax(output.logits[:, -1, :], axis=-1)
                mx.eval(token, output.hidden_states, [entry.state for entry in cache])
            stream = self._run_rounds(
                self._model, self._draft, cache, full_input_ids, token, None, output,
                draft_kind="mtp", max_tokens=max_new_tokens,
                sampler=lambda logits: mx.argmax(logits, axis=-1),
                draft_block_size=self.block_size, sampler_is_greedy=True,
            )
            first, _ = next(stream)
            if type(first) is not int or first != int(token.item()):
                raise RuntimeError("MTP did not yield the target's first token at the prefill boundary")
            if self._draft.accept_lens or self._draft.draft_lens:
                raise RuntimeError("MTP speculative work leaked into prefill")
            generated.append(first)
            mx.synchronize()
            prefill_seconds = perf_counter() - started
            decode_started = perf_counter()
            if max_new_tokens > 1:
                for next_token, _ in stream:
                    if type(next_token) is not int or next_token < 0:
                        raise RuntimeError("MTP yielded an invalid token ID")
                    generated.append(next_token)
                    if len(generated) > max_new_tokens:
                        raise RuntimeError("MTP exceeded the requested output budget")
            if len(generated) != max_new_tokens:
                raise RuntimeError("MTP did not complete the requested fixed output budget")
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                try:
                    # Committed recurrent/KV states and the final drafter seed
                    # can remain lazy after the last yielded token.
                    mx.eval([entry.state for entry in cache], self._draft.draft_eval_state())
                finally:
                    mx.synchronize()
        decode_seconds = perf_counter() - decode_started if max_new_tokens > 1 else 0.0
        counters = _acceptance_metrics(self._draft.accept_lens, self._draft.draft_lens, len(generated))
        result = {
            "prompt_tokens": len(prompt_tokens), "generated_tokens": len(generated),
            "decode_tokens": len(generated) - 1,
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "generated_token_ids": generated,
            "output_text": self.tokenizer.decode(generated, skip_special_tokens=False),
            "peak_memory_gb": float(mx.get_peak_memory()) / 1_000_000_000,
            "timing_method": "perf_counter + MLX phase synchronization; full prompt hidden-state capture and first target token in prefill; MTP head prefill, stock round loop and final cache materialization in decode",
        }
        result.update(counters)
        return result

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        quantization = self._draft_config.get("quantization") or self._draft_config.get("quantization_config") or {}
        metadata.update({
            "framework": "mlx-vlm-mtp", "target_framework": "mlx-vlm",
            "algorithm": "native Qwen3.5 multi-token prediction with stock MLX-VLM target verification",
            "upstream_repository": "https://github.com/Blaizzy/mlx-vlm",
            "upstream_runtime_sources": self._runtime_sources,
            "draft_model_path": self.draft_path,
            "draft_repo_id": self._draft_manifest.get("repo_id"),
            "draft_revision": self._draft_manifest.get("resolved_revision"),
            "draft_manifest_sha256": self._draft_manifest_sha256,
            "draft_weight_bits": quantization.get("bits"),
            "draft_weight_group_size": quantization.get("group_size"),
            "draft_quantization_strategy": "checkpoint-provided quantization loaded unchanged by mlx_vlm.speculative.drafters.load_drafter",
            "block_size": self.block_size, "max_draft_tokens_per_round": self.block_size - 1,
            "measurement_harness": "phase-separated native MTP with full prompt hidden capture",
            "prefill_hidden_capture": "normalized target hidden states retained for every prompt token across chunks; original full prompt passed to stock run_speculative_rounds",
            "draft_prefill_phase": "decode, after the first target token is available",
            "acceptance_accounting": "stock accept_lens/draft_lens; all verified matching drafts including capped final proposals, bonus excluded",
            "timed_text_detokenization": False,
            "greedy_parity": "must be validated against same-environment MLXVLMBackend",
        })
        return metadata
