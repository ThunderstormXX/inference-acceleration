"""Calibration-only vocabulary shortlist for native MTP proposals.

Target verification and correction always retain the original full vocabulary.
A restricted draft may accept fewer tokens; it never restricts the target's
output vocabulary. This module imports no MLX until an adapter is constructed.
"""
from __future__ import annotations

from hashlib import sha256
from contextlib import contextmanager
import json
from pathlib import Path
from threading import RLock


_CALIBRATION_ROWS = tuple(range(100, 110))
_VOCAB_LOCK = RLock()


def build_draft_vocabulary(bundle, tokenizer, model_config, target_manifest_sha256, *, prefix_tokens=0):
    """Use only generated output IDs from rows 100..109 plus declared specials."""
    vocab_size = model_config.get("text_config", model_config)["vocab_size"]
    if type(vocab_size) is not int or vocab_size < 2:
        raise ValueError("Invalid model vocabulary size")
    if type(prefix_tokens) is not int or not 0 <= prefix_tokens <= vocab_size:
        raise ValueError("prefix_tokens must be an integer between zero and target vocabulary size")
    selected = [row for row in bundle["chains"] if row.get("source_row_index") in _CALIBRATION_ROWS]
    if sorted(row["source_row_index"] for row in selected) != list(_CALIBRATION_ROWS):
        raise ValueError("Need exactly one complete calibration chain for each row 100..109")
    output_ids = set()
    for row in selected:
        tokens = row["generated_token_ids"]
        if row.get("source_method") != "confidence_collect" or len(tokens) != 2048:
            raise ValueError("Expected completed 2048-token calibration confidence trajectories")
        if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
            raise ValueError("Invalid calibration output token")
        output_ids.update(tokens)
    specials = sorted({item["id"] for item in tokenizer.get("added_tokens", []) if item.get("special") is True})
    if any(type(token) is not int or not 0 <= token < vocab_size for token in specials):
        raise ValueError("Invalid tokenizer special token ID")
    calibration_sources = [source for source in bundle.get("provenance", [])
                           if "/calibration2048/" in source.get("path", "")]
    if (len(calibration_sources) != 1
            or calibration_sources[0].get("target_manifest_sha256") != target_manifest_sha256):
        raise ValueError("Calibration provenance does not match the target checkpoint")
    ids = sorted(output_ids | set(specials) | set(range(prefix_tokens)))
    result = {"schema_version": 1, "kind": "mtp_draft_only_vocabulary",
            "target_manifest_sha256": target_manifest_sha256,
            "target_vocab_size": vocab_size, "token_ids": ids,
            "calibration": {"source_row_indices": list(_CALIBRATION_ROWS), "chains": 10,
                            "generated_tokens_per_chain": 2048, "total_generated_tokens": 20480,
                            "distinct_output_tokens": len(output_ids), "source": calibration_sources[0]},
            "special_token_ids": specials, "shortlist_size": len(ids),
            "vocabulary_fraction": len(ids) / vocab_size,
            "selection": "Union of calibration output token IDs and tokenizer-declared special IDs; no prompt IDs or evaluation outputs",
            "evaluation_exclusions": {"profiling_prompt_indices": [0, 1, 2, 3, 4], "heldout_source_row_indices": [110]},
            "limitation": "These are previously inspected development chains; disjoint evaluation, not a claim of a pristine test set."}
    if prefix_tokens:
        result["capacity"] = {"prefix_token_count": prefix_tokens,
                              "rule": "Include every token ID in range(0, prefix_token_count), independent of evaluation outputs"}
        result["selection"] = (f"Deterministic token ID prefix 0..{prefix_tokens - 1}, union calibration output IDs "
                               "and tokenizer-declared special IDs; no prompt IDs or evaluation outputs")
        result["development_tuning"] = ("Vocabulary capacity variant chosen after inspecting development benchmark "
                                        "coverage; no evaluation token IDs were added individually. This is not pristine validation.")
    return result


def load_draft_vocabulary(path):
    payload = Path(path).read_bytes()
    config = json.loads(payload)
    ids = config.get("token_ids")
    size = config.get("target_vocab_size")
    if (config.get("schema_version") != 1 or config.get("kind") != "mtp_draft_only_vocabulary"
            or type(size) is not int or size < 2 or not isinstance(ids, list) or not ids
            or any(type(token) is not int or not 0 <= token < size for token in ids)
            or ids != sorted(set(ids)) or config.get("shortlist_size") != len(ids)
            or config.get("calibration", {}).get("source_row_indices") != list(_CALIBRATION_ROWS)):
        raise ValueError("Malformed or incompatible draft vocabulary artifact")
    return config, sha256(payload).hexdigest()


class ScopedDraftVocabulary:
    """Replace only ordinary native MTP's greedy draft readout, reversibly."""
    def __init__(self, backend, vocabulary_path, *, quantized_linear_class=None):
        self.backend = backend
        self.path = Path(vocabulary_path).resolve()
        self.config, self.config_sha256 = load_draft_vocabulary(self.path)
        self.draft = getattr(backend, "_draft", None)
        self._validate_backend()
        if quantized_linear_class is None:
            from mlx.nn import QuantizedLinear
            quantized_linear_class = QuantizedLinear
        lm = getattr(backend._model, "language_model", backend._model)
        self.head = getattr(lm, "lm_head", None)
        if not isinstance(self.head, quantized_linear_class):
            raise ValueError("Draft shortlist requires an explicit quantized shared LM head")
        if self.head.mode != "affine" or self.head.biases is None:
            raise ValueError("Draft shortlist currently supports affine quantization only")
        if int(self.head.weight.shape[0]) != self.config["target_vocab_size"]:
            raise ValueError("Draft vocabulary size does not match target LM head")
        model_path = Path(backend.model_path)
        if sha256((model_path / "download-manifest.json").read_bytes()).hexdigest() != self.config["target_manifest_sha256"]:
            raise ValueError("Draft vocabulary belongs to a different target checkpoint")
        tokenizer_sha256 = self.config.get("tokenizer_sha256")
        if (not tokenizer_sha256
                or sha256((model_path / "tokenizer.json").read_bytes()).hexdigest() != tokenizer_sha256):
            raise ValueError("Draft vocabulary belongs to a different tokenizer")
        self.mx = backend._mx
        self.ids = self.mx.array(self.config["token_ids"], dtype=self.mx.int32)
        # Gather packed quantized rows unchanged. There is no dequantization,
        # retraining or requantization, and the original shared head is intact.
        self.weight = self.mx.take(self.head.weight, self.ids, axis=0)
        self.scales = self.mx.take(self.head.scales, self.ids, axis=0)
        self.biases = self.mx.take(self.head.biases, self.ids, axis=0)
        self.bias = self.mx.take(self.head["bias"], self.ids, axis=0) if "bias" in self.head else None
        self.mx.eval(self.ids, self.weight, self.scales, self.biases, *([] if self.bias is None else [self.bias]))
        self.mx.synchronize()
        self.calls = 0
        self._active = False
        self._original = None
        self._owned = False
        self._suspended = False

    def _validate_backend(self):
        if hasattr(self.backend, "_confidence_records"):
            raise ValueError("Draft shortlist cannot be combined with confidence collection")
        if self.draft is None or type(self.draft).__name__ != "Qwen3_5MTPDraftModel":
            raise ValueError("Draft shortlist requires the ordinary native Qwen3.5 MTP drafter")
        if not callable(getattr(self.draft, "_greedy_token", None)):
            raise ValueError("Native MTP greedy readout API is unavailable")
        if self.backend._model is None or self.backend._mx is None:
            raise ValueError("Load the target and draft before constructing a shortlist")

    def _greedy_token(self, hidden):
        self._validate_backend()
        logits = self.mx.quantized_matmul(hidden, self.weight, scales=self.scales,
            biases=self.biases, transpose=True, group_size=self.head.group_size,
            bits=self.head.bits, mode=self.head.mode)
        if self.bias is not None:
            logits = logits + self.bias
        self.calls += 1
        return self.mx.take(self.ids, self.mx.argmax(logits, axis=-1), axis=0)

    def __enter__(self):
        _VOCAB_LOCK.acquire()
        try:
            if self._active:
                raise RuntimeError("Draft vocabulary context is already active")
            self._validate_backend()
            if "_greedy_token" in vars(self.draft):
                raise ValueError("Another observer already overrides the MTP greedy readout")
            self._owned = "_greedy_token" in vars(self.draft)
            self._original = vars(self.draft).get("_greedy_token")
            self.draft._greedy_token = self._greedy_token
            self._active = True
            return self
        except BaseException:
            _VOCAB_LOCK.release()
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self._owned:
                self.draft._greedy_token = self._original
            else:
                delattr(self.draft, "_greedy_token")
            self._active = False
        finally:
            _VOCAB_LOCK.release()
        return False

    @contextmanager
    def suspended(self):
        """Run stock draft readout while keeping all shortlist arrays resident."""
        if not self._active or self._suspended:
            raise RuntimeError("Suspend requires one active, unsuspended draft vocabulary context")
        self._suspended = True
        if self._owned:
            self.draft._greedy_token = self._original
        else:
            delattr(self.draft, "_greedy_token")
        try:
            yield self
        finally:
            self.draft._greedy_token = self._greedy_token
            self._suspended = False

    def metadata(self):
        return {"optimization": "calibration vocabulary for draft readout only",
                "vocabulary_path": str(self.path), "vocabulary_sha256": self.config_sha256,
                "shortlist_size": len(self.config["token_ids"]),
                "target_vocab_size": self.config["target_vocab_size"],
                "calibration_source_rows": self.config["calibration"]["source_row_indices"],
                "capacity": self.config.get("capacity", {"prefix_token_count": 0}),
                "development_tuning": self.config.get("development_tuning"),
                "draft_readout_calls": self.calls,
                "head_bits": self.head.bits, "head_group_size": self.head.group_size,
                "weight_handling": "selected packed rows gathered unchanged; no requantization",
                "target_verification": "unchanged full vocabulary and original correction/bonus sampling",
                "draft_distribution": "restricted greedy argmax; acceptance can differ",
                "trace_compatibility": "standard MTP observer records valid global proposal IDs; no confidence probabilities claimed",
                "timing": "shortlist materialized before warmup; readout and ID mapping included in decode",
                "token_parity": "must be measured against full-vocabulary stock AR"}
