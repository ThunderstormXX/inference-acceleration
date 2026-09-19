"""Phase-separated adapter around the pinned, official Z-Lab DFlash runtime.

Upstream yields a block before rewinding its caches, and its ``accepted`` field
includes the target bonus. A scoped observer records actual rollback arguments
and the adapter exhausts the stream before timing stops. No upstream math changes.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution, version
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from threading import RLock
from time import perf_counter
from typing import Any

from ..mlx_backend import MLXBackend


UPSTREAM_COMMIT = "07ebd93db9f472af339b644bb70221ad8428328a"
_UPSTREAM_LOCK = RLock()


@contextmanager
def _temporary_attribute(owner: Any, name: str, value: Any):
    previous = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, previous)


class _NoOpDetokenizer:
    """Keep output string work outside both measured phases."""

    def __init__(self, tokenizer: Any):
        self.last_segment = ""

    def add_token(self, token: int) -> None:
        pass

    def finalize(self) -> None:
        pass


class _RoundObserver:
    """Resolve a yielded round after upstream has committed or rewound it."""

    def __init__(self, block_size: int, max_tokens: int):
        self.block_size = block_size
        self.max_tokens = max_tokens
        self.rounds: list[dict[str, int]] = []
        self.pending: dict[str, int] | None = None
        self.caches: list[list[Any]] = []

    def yielded(self, emitted_before: int, emitted: int) -> None:
        self.finish_pending()
        drafted = min(self.block_size, self.max_tokens - emitted_before + 1) - 1
        if not 1 <= emitted <= min(drafted + 1, self.max_tokens - emitted_before):
            raise RuntimeError("Unexpected DFlash block length")
        self.pending = {"drafted_tokens": drafted, "emitted_tokens": emitted}

    def rollback(self, accepted: int, trimmed: int) -> None:
        if self.pending is None:
            raise RuntimeError("DFlash rollback occurred without a yielded round")
        if "accepted_draft_tokens" in self.pending:
            raise RuntimeError("DFlash rolled back the same round more than once")
        drafted = self.pending["drafted_tokens"]
        if not 0 <= accepted < drafted or trimmed != drafted - accepted:
            raise RuntimeError("Unexpected DFlash rollback accounting")
        self.pending["accepted_draft_tokens"] = accepted

    def finish_pending(self) -> None:
        if self.pending is None:
            return
        row = self.pending
        # On our supported hybrid target every rejection calls rollback. No
        # rollback after resuming therefore means the entire draft was accepted.
        accepted = row.setdefault("accepted_draft_tokens", row["drafted_tokens"])
        if row["emitted_tokens"] != accepted + 1 and not (
            accepted == row["drafted_tokens"] == row["emitted_tokens"]
        ):
            raise RuntimeError("DFlash emitted tokens disagree with verified acceptance")
        row["emitted_draft_tokens"] = min(accepted, row["emitted_tokens"])
        row["emitted_target_tokens"] = row["emitted_tokens"] - row["emitted_draft_tokens"]
        self.rounds.append(row)
        self.pending = None

    def metrics(self) -> dict[str, Any]:
        if self.pending is not None:
            raise RuntimeError("DFlash stream was not exhausted")
        drafted = sum(row["drafted_tokens"] for row in self.rounds)
        accepted = sum(row["accepted_draft_tokens"] for row in self.rounds)
        return {
            "drafted_tokens": drafted,
            "accepted_draft_tokens": accepted,
            "draft_acceptance_rate": accepted / drafted if drafted else None,
            "speculative_rounds": len(self.rounds),
            "mean_accepted_draft_tokens_per_round": accepted / len(self.rounds) if self.rounds else None,
            "emitted_draft_tokens": sum(row["emitted_draft_tokens"] for row in self.rounds),
            "emitted_target_tokens": 1 + sum(row["emitted_target_tokens"] for row in self.rounds),
            "speculative_round_details": self.rounds,
        }


class DFlashBackend(MLXBackend):
    """Official DFlash proposals verified by the unchanged MLX-LM target.

Only batch-one Qwen3.5 hybrid targets are supported. All target outputs remain
greedy and EOS is ignored for the benchmark's fixed output budget. Token parity
must be checked by the paired runner; this class does not claim empirical parity.
"""

    def __init__(
        self,
        model_path: str,
        draft_path: str,
        prefill_step_size: int = 512,
        block_size: int = 5,
        draft_bits: int | None = 4,
        source_path: str | None = None,
        kv_bits: int | None = None,
        wired_memory: bool = True,
    ) -> None:
        if kv_bits is not None:
            raise ValueError("Official DFlash adapter requires an unquantized KV cache")
        if wired_memory is not True:
            raise ValueError("Official DFlash uses the recommended wired-memory limit; use wired_memory=True")
        if type(block_size) is not int or not 2 <= block_size <= 5:
            raise ValueError("block_size must be between 2 and 5 for this quantized-target experiment")
        if draft_bits is not None and (type(draft_bits) is not int or draft_bits not in (4, 8)):
            raise ValueError("draft_bits must be None, 4, or 8")
        super().__init__(model_path, prefill_step_size, kv_bits, wired_memory)
        self.draft_model_path = str(Path(draft_path).expanduser().resolve())
        self.source_path = str(Path(source_path).expanduser().resolve()) if source_path else None
        self.block_size = block_size
        self.draft_bits = draft_bits
        self._draft: Any = None
        self._upstream: Any = None
        self._timing_tokenizer: Any = None
        self._source_sha256: str | None = None
        self._draft_config: dict[str, Any] = {}
        self._draft_manifest: dict[str, Any] = {}
        self._draft_manifest_sha256: str | None = None

    def _source_file(self) -> Path:
        if self.source_path is None:
            runtime = distribution("dflash")
            direct_url = json.loads(runtime.read_text("direct_url.json") or "{}")
            revision = direct_url.get("vcs_info", {}).get("commit_id")
            if revision != UPSTREAM_COMMIT:
                raise ValueError(f"Installed DFlash must be pinned to {UPSTREAM_COMMIT}, got {revision}")
            return Path(runtime.locate_file("dflash/model_mlx.py"))
        source_root = Path(self.source_path)
        source = source_root / "dflash/model_mlx.py"
        if not source.is_file():
            raise FileNotFoundError(f"Official DFlash source missing: {source}")
        revision = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != UPSTREAM_COMMIT:
            raise ValueError(f"DFlash source must be pinned to {UPSTREAM_COMMIT}, got {revision}")
        checked_in = subprocess.check_output(
            ["git", "-C", str(source_root), "show", "HEAD:dflash/model_mlx.py"]
        )
        if source.read_bytes() != checked_in:
            raise ValueError("Official DFlash model_mlx.py has uncommitted modifications")
        return source

    def _load_upstream(self) -> Any:
        source = self._source_file()
        source_bytes = source.read_bytes()
        self._resolved_source_path = str(source.resolve())
        self._source_sha256 = sha256(source_bytes).hexdigest()
        module_name = "inference_lab_official_dflash_" + self._source_sha256[:12]
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load DFlash source at {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return module

    def load(self) -> DFlashBackend:
        if self._model is not None and self._draft is not None:
            return self
        target_config = json.loads((Path(self.model_path) / "config.json").read_text())
        text_config = target_config.get("text_config", target_config)
        if target_config.get("model_type") != "qwen3_5" or "linear_attention" not in text_config.get("layer_types", []):
            raise ValueError("DFlash acceptance observer supports only Qwen3.5 hybrid targets")
        draft_path = Path(self.draft_model_path)
        manifest_bytes = (draft_path / "download-manifest.json").read_bytes()
        self._draft_manifest = json.loads(manifest_bytes)
        if not self._draft_manifest.get("repo_id") or not self._draft_manifest.get("resolved_revision"):
            raise ValueError("Draft download manifest must identify its repository and revision")
        self._draft_manifest_sha256 = sha256(manifest_bytes).hexdigest()
        self._draft_config = json.loads((draft_path / "config.json").read_text())
        for key in ("hidden_size", "num_target_layers", "vocab_size"):
            target_key = "num_hidden_layers" if key == "num_target_layers" else key
            if self._draft_config.get(key) != text_config.get(target_key):
                raise ValueError(f"Draft/target {key} mismatch")
        if "DFlashDraftModel" not in self._draft_config.get("architectures", []):
            raise ValueError("This adapter requires the Qwen3.5 DFlash drafter architecture")
        with _UPSTREAM_LOCK:
            self._upstream = self._load_upstream()
            # Upstream only accepts Hub ids; use its unmodified loader with a
            # scoped local resolver. No network access occurs in this adapter.
            with _temporary_attribute(self._upstream, "snapshot_download", lambda *args, **kwargs: str(draft_path)):
                self._draft = self._upstream.load_draft(self.draft_model_path)
            if self.draft_bits is not None:
                self._upstream.nn.quantize(self._draft, group_size=64, bits=self.draft_bits)
            self._upstream.mx.eval(self._draft.parameters())
            self._upstream.mx.synchronize()
            self._upstream.mx.clear_cache()
            # Quantize the small drafter before loading the 9B target, avoiding
            # simultaneous full-precision and quantized drafter allocations.
            super().load()
            raw_tokenizer = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
            self._timing_tokenizer = self._upstream.TokenizerWrapper(
                raw_tokenizer, detokenizer_class=_NoOpDetokenizer, eos_token_ids=[]
            )
        return self

    @contextmanager
    def _observe(self, observer: _RoundObserver):
        upstream = self._upstream
        original_capture = upstream._GDNStateCapture
        original_make_cache = upstream.make_prompt_cache

        class ObservedCapture(original_capture):
            def rollback(capture, cache, accepted, trim):
                observer.rollback(int(accepted), int(trim))
                return super().rollback(cache, accepted, trim)

        def observed_make_cache(model):
            cache = original_make_cache(model)
            if model is self._model and upstream.can_trim_prompt_cache(cache):
                raise RuntimeError("Expected a non-trimmable Qwen3.5 recurrent cache")
            observer.caches.append(cache)
            return cache

        with _temporary_attribute(upstream, "_GDNStateCapture", ObservedCapture):
            with _temporary_attribute(upstream, "make_prompt_cache", observed_make_cache):
                yield

    def measure(self, prompt_tokens: list[int], max_new_tokens: int = 128) -> dict[str, Any]:
        if self._model is None or self._draft is None:
            raise RuntimeError("Call load() before measure()")
        if not prompt_tokens or any(type(token) is not int or token < 0 for token in prompt_tokens):
            raise ValueError("prompt_tokens must contain nonnegative integer token IDs")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        with _UPSTREAM_LOCK:
            return self._measure_speculative(prompt_tokens, max_new_tokens)

    def _measure_speculative(self, prompt_tokens: list[int], max_new_tokens: int) -> dict[str, Any]:
        mx = self._mx
        observer = _RoundObserver(self.block_size, max_new_tokens)
        prompt = mx.array(prompt_tokens, dtype=mx.uint32)
        mx.eval(prompt)
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        generated: list[int] = []
        stream = None
        # Same upstream wired context as stream_generate, outside phase timers
        # to match the ordinary MLXBackend's memory-policy accounting.
        with self._upstream.wired_limit(self._model, [self._upstream.generation_stream]):
            with self._observe(observer):
                try:
                    stream = self._upstream._stream_generate(
                        self._model, self._draft, self._timing_tokenizer, prompt,
                        block_size=self.block_size, max_tokens=max_new_tokens,
                        temperature=0.0, top_p=1.0, top_k=0,
                        prefill_step_size=self.prefill_step_size,
                    )
                    started = perf_counter()
                    first = next(stream)
                    if len(first.tokens) != 1 or first.accepted is not None or first.finish_reason is not None:
                        raise RuntimeError("Expected exactly one target token at the prefill boundary")
                    generated.extend(first.tokens)
                    mx.synchronize()
                    prefill_seconds = perf_counter() - started
                    decode_started = perf_counter()
                    saw_final = False
                    for response in stream:
                        # Resuming upstream has now applied the previous
                        # round's rewind, even for a length-capped final block.
                        observer.finish_pending()
                        if response.tokens:
                            if saw_final or response.finish_reason is not None or response.accepted is None:
                                raise RuntimeError("Unexpected early stop or block metadata in DFlash")
                            observer.yielded(len(generated), len(response.tokens))
                            generated.extend(response.tokens)
                        else:
                            if response.finish_reason != "length" or saw_final:
                                raise RuntimeError("Unexpected DFlash terminal response")
                            saw_final = True
                    if not saw_final or len(generated) != max_new_tokens:
                        raise RuntimeError("DFlash did not complete the requested fixed output budget")
                finally:
                    try:
                        if stream is not None:
                            stream.close()
                    finally:
                        try:
                            # Rollback states are lazy MLX arrays. synchronize
                            # alone does not materialize them; evaluate first.
                            mx.eval([entry.state for cache in observer.caches for entry in cache])
                        finally:
                            try:
                                mx.synchronize()
                            finally:
                                if hasattr(self._model, "_hidden_states"):
                                    self._model._hidden_states[:] = [None] * len(self._model._hidden_states)
                decode_seconds = perf_counter() - decode_started if max_new_tokens > 1 else 0.0
        result = {
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": len(generated),
            "decode_tokens": len(generated) - 1,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "generated_token_ids": generated,
            "output_text": self.tokenizer.decode(generated, skip_special_tokens=False),
            "peak_memory_gb": float(mx.get_peak_memory()) / 1_000_000_000,
            "timing_method": "perf_counter + MLX synchronization; first target token in prefill; full DFlash stream, final rollback and cache evaluation in decode",
        }
        result.update(observer.metrics())
        return result

    def metadata(self) -> dict[str, Any]:
        result = super().metadata()
        try:
            result["versions"]["dflash"] = version("dflash")
        except PackageNotFoundError:
            result["versions"]["dflash"] = None
        result.update({
            "framework": "dflash-mlx",
            "target_framework": "mlx-lm",
            "algorithm": "DFlash block-diffusion speculative decoding",
            "upstream_repository": "https://github.com/z-lab/dflash",
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source_sha256": self._source_sha256,
            "upstream_source_path": getattr(self, "_resolved_source_path", self.source_path),
            "draft_model_path": self.draft_model_path,
            "draft_repo_id": self._draft_manifest.get("repo_id"),
            "draft_revision": self._draft_manifest.get("resolved_revision"),
            "draft_manifest_sha256": self._draft_manifest_sha256,
            "draft_weight_bits": self.draft_bits,
            "draft_weight_group_size": 64 if self.draft_bits is not None else None,
            "draft_quantization_strategy": (
                "runtime mlx.nn.quantize affine, group_size=64, before binding shared target embedding/head; original downloaded weights preserved"
                if self.draft_bits is not None else "unquantized original draft weights"
            ),
            "block_size": self.block_size,
            "max_draft_tokens_per_round": self.block_size - 1,
            "acceptance_accounting": "all verified matching drafts, including non-emitted final proposals; exact rollback observer; target bonus excluded",
            "timed_text_detokenization": False,
            "greedy_parity": "must be independently validated against same-environment MLXBackend",
            "wired_memory_policy": "official mlx_lm.generate.wired_limit, outside phase timers; restored on exit",
        })
        return result
