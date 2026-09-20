"""Instrumented native-MTP confidence gating with stock target transactions.

The gate runs after proposals are computed. Its one-token fallback retains the
MTP cache/seed maintenance cost and is not the ordinary autoregressive backend.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import importlib
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from .mtp_backend import MTPBackend


class ConfidenceMTPBackend(MTPBackend):
    """Collect normalized proposal probabilities or gate a two-token draft.

    Greedy proposal IDs keep the original fused-argmax path. Probabilities use
    an additional full-vocabulary projection, then float32 log-sum-exp; this
    observer work is included in the measured decode duration. Only greedy
    target parity is supported; no stochastic distribution-preservation claim.
    """

    def __init__(self, model_path: str, draft_path: str,
                 prefill_step_size: int = 512, block_size: int = 3,
                 wired_memory: bool = True, *, threshold: float = 0.5,
                 policy: str = "collect") -> None:
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be finite and between 0 and 1")
        if policy not in ("collect", "gate", "no_draft"):
            raise ValueError("Confidence policy must be collect, gate or no_draft")
        if type(block_size) is not int or block_size != 3:
            raise ValueError("Confidence MTP requires block_size=3 (two proposals)")
        super().__init__(model_path, draft_path, prefill_step_size, block_size, wired_memory)
        self.threshold = float(threshold)
        self.policy = policy
        self.trace_generation = True
        self._mtp_module: Any = None
        self._confidence_records: list[dict[str, Any]] = []
        self._confidence_recorder: Any = None

    def load(self):
        super().load()
        self._mtp_module = importlib.import_module("mlx_vlm.speculative.mtp")
        if not callable(getattr(self._model, "speculative_logits_from_hidden", None)):
            raise RuntimeError("Confidence MTP needs the Qwen3.5 hidden-to-logits projection")
        self._run_rounds = self._confidence_rounds
        for source in (Path(__file__), Path(importlib.import_module(type(self._model).__module__).__file__)):
            self._runtime_sources[str(source.resolve())] = {
                "path": str(source.resolve()), "sha256": sha256(source.read_bytes()).hexdigest(),
            }
        return self

    def _measure(self, prompt_tokens, max_new_tokens=128):
        if self._model is None or self._draft is None or self._mtp_module is None:
            raise RuntimeError("Call load() before measure()")
        if not self.trace_generation:
            raise ValueError("Confidence MTP requires trace_generation=True")
        if not prompt_tokens or any(type(token) is not int or token < 0 for token in prompt_tokens):
            raise ValueError("prompt_tokens must contain nonnegative integer token IDs")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        from inference_lab.visualization.recording import GenerationRecorder
        self._confidence_records = []
        recorder = GenerationRecorder(self, clock=perf_counter, speculative=True)
        self._confidence_recorder = recorder
        self._draft.reset(self._model)
        try:
            result = self._measure_mtp(prompt_tokens, max_new_tokens, recorder=recorder)
            result["timing_method"] += "; probability projection/host-read and gate decision included in decode"
            result["generation_trace"]["instrumentation"].update({
                "confidence_policy": self.policy,
                "confidence_overhead": "additional full vocabulary readout, float32 logsumexp and scalar host read included",
                "gate_timing": "after proposal generation; generated proposals discarded by gate are not target rejections",
                "fallback": "one-token target transaction plus native MTP cache/seed maintenance",
            })
            return result
        finally:
            self._draft.reset(self._model)
            self._confidence_recorder = None

    @contextmanager
    def _proposal_hidden_capture(self):
        """Observe the existing argmax calls, preserving inherited lookup."""
        draft = self._draft
        had_local = "_greedy_token" in vars(draft)
        original = draft._greedy_token
        hidden_states = []

        def observe(hidden):
            token = original(hidden)
            hidden_states.append(hidden)
            return token

        draft._greedy_token = observe
        try:
            yield hidden_states
        finally:
            if had_local:
                draft._greedy_token = original
            else:
                del draft._greedy_token

    def _proposal_logprobs(self, hidden_states, proposal_tokens):
        mx = self._mx
        with mx.stream(self._generation_stream):
            hidden = mx.concatenate(hidden_states, axis=1)
            logits = self._model.speculative_logits_from_hidden(hidden).astype(mx.float32)
            selected = mx.take_along_axis(logits, proposal_tokens[..., None], axis=-1)[..., 0]
            logprobs = selected - mx.logsumexp(logits, axis=-1)
        # Transfer only normalized scalar scores, never vocabulary arrays.
        values = [float(value) for value in logprobs.reshape(-1).tolist()]
        if any(not math.isfinite(value) or value > 1e-6 for value in values):
            raise RuntimeError("Invalid normalized MTP proposal log-probability")
        return [min(0.0, value) for value in values]

    def _confidence_rounds(self, model, draft, cache, input_ids, first_token,
                           logprobs, last_outputs, *, draft_kind, max_tokens,
                           sampler, draft_block_size=None, sampler_is_greedy=False):
        if draft_kind != "mtp" or not sampler_is_greedy or input_ids.shape[0] != 1:
            raise ValueError("Confidence MTP supports single-request greedy native MTP only")
        if draft_block_size != 3:
            raise ValueError("Confidence round loop requires block_size=3")
        mx, module, recorder = self._mx, self._mtp_module, self._confidence_recorder
        module._buffer_mtp_target_cache(cache, draft, draft_block_size)
        mx.eval(first_token)
        bonus = int(first_token.item())
        yield bonus, logprobs  # The parent's unchanged prefill boundary.
        if max_tokens <= 1:
            return
        draft.reset(model)
        rng = module._SpeculativeSamplerRNG(draft, enabled=False)
        hidden = last_outputs.hidden_states[-1]
        rng.draft_call(draft.prefill_from_target_hidden, input_ids, hidden, bonus,
                       sampler, input_ids.dtype, greedy=True)
        hidden = module._mtp_draft_hidden(model, hidden[:, -1:, :])
        offset = module._mtp_cache_offset_max(cache)
        draft.set_shared_kv(last_outputs.shared_kv_states, offset,
                            position=module._mtp_draft_position(offset), kv_valid_len=offset)
        emitted, round_id = 1, 0
        while emitted < max_tokens:
            round_id += 1
            started = recorder.now()
            proposals = mx.array([[]], dtype=input_ids.dtype)
            ids, logps = [], []
            if self.policy != "no_draft":
                seed_hidden = draft._seed_hidden
                if seed_hidden is None or draft._seed_token is None:
                    raise RuntimeError("Native MTP did not provide an aligned proposal seed")
                block_total = min(3, max_tokens - emitted + 1)
                with self._proposal_hidden_capture() as later_hidden:
                    proposals = rng.draft_tokens(draft.draft_block, bonus, hidden, None,
                                                 block_total, sampler, input_ids.dtype, greedy=True)
                ids = [int(token) for token in proposals.reshape(-1).tolist()]
                draft_ready = recorder.now()
                hidden_states = [seed_hidden, *later_hidden]
                if len(hidden_states) != len(ids) or len(ids) != block_total - 1:
                    raise RuntimeError("MTP proposal hidden states are not aligned with chosen tokens")
                logps = self._proposal_logprobs(hidden_states, proposals)
                if len(logps) != len(ids):
                    raise RuntimeError("MTP confidence count differs from proposal count")
                probabilities_ready = recorder.now()
            else:
                # An unused precomputed seed is replaced by accept_verified_tokens.
                # There were no speculative cache appends to keep or trim.
                draft._round_appended = 0
                draft_ready = probabilities_ready = recorder.now()
            probabilities = [math.exp(value) for value in logps]
            log_product = sum(logps) if logps else None
            product = math.exp(log_product) if log_product is not None else None
            fallback = self.policy == "no_draft" or (
                self.policy == "gate" and self.threshold > 0 and log_product < math.log(self.threshold)
            )
            verified_tokens = mx.array([[]], dtype=input_ids.dtype) if fallback else proposals
            verified_count = 0 if fallback else len(ids)
            if not fallback:
                recorder.events.append({"type": "draft", "t": draft_ready,
                                        "token_ids": ids, "output_count": emitted, "round": round_id})
            verify = None
            verify_started = recorder.now()
            try:
                with mx.stream(self._generation_stream):
                    verify_input = mx.concatenate([mx.array([[bonus]], dtype=input_ids.dtype), verified_tokens], axis=1)
                    verify = module._mtp_verify_target(model, verify_input, cache, sampler,
                                                       sample_target_tokens=True)
                accepted, new_tokens = module._mtp_acceptance_walk(
                    model, verify, verified_tokens, sampler, max_tokens - emitted,
                    row_id=0, base_position=emitted,
                )
                if verify.target_tokens is None:
                    raise RuntimeError("Confidence MTP requires materialized greedy target IDs")
                target_ids = [int(token) for token in verify.target_tokens.reshape(-1).tolist()]
                verification_done = recorder.now()
                if fallback and (accepted != 0 or len(new_tokens) != 1):
                    raise RuntimeError("Anchor-only verification must emit exactly one target token")
                if not fallback:
                    module._record_speculative_round(draft, accepted, verified_count)
                # This is the native trim/replay path even when the gate rejects
                # all proposals: trim _round_appended, then replay the new bonus.
                rng.draft_call(draft.accept_verified_tokens, verify.hidden, verified_tokens,
                               accepted, new_tokens, sampler, input_ids.dtype, greedy=True)
                verify.commit(model, cache, accepted, verified_count + 1)
                committed = recorder.now()
            except BaseException:
                if verify is not None:
                    verify.abort()
                abort_draft = getattr(draft, "abort_draft_round", None)
                if callable(abort_draft):
                    abort_draft()
                raise
            timing = {
                "round_started_t": started, "draft_ready_t": draft_ready,
                "probabilities_ready_t": probabilities_ready,
                "verification_started_t": verify_started,
                "verification_completed_t": verification_done,
                "cache_commit_enqueued_t": committed,
                "draft_seconds": draft_ready - started,
                "probability_seconds": probabilities_ready - draft_ready,
                "verify_seconds": verification_done - verify_started,
                "commit_enqueue_seconds": committed - verification_done,
            }
            self._confidence_records.append({
                "round": round_id, "output_count_before": emitted,
                "proposal_token_ids": ids, "proposal_logprobs": logps,
                "proposal_probabilities": probabilities,
                "p1": probabilities[0] if probabilities else None,
                "p2": probabilities[1] if len(probabilities) > 1 else None,
                "confidence_product": product, "log_confidence_product": log_product,
                "decision": "fallback" if fallback else "verify",
                "fallback_reason": self.policy if fallback else None,
                "verified_draft_count": verified_count,
                "accepted_count": None if fallback else int(accepted),
                "target_token_ids": target_ids, "emitted_token_ids": list(new_tokens),
                "timing": timing,
            })
            fields = {
                "t": committed, "round": 0 if fallback else round_id,
                "confidence_round": round_id, "accepted_count": int(accepted),
                "draft_count": verified_count,
                "proposed_token_ids": [] if fallback else ids,
                "rejected_token_ids": [] if fallback else ids[accepted:],
                "target_token_ids": target_ids,
                "verification_completed_t": verification_done,
                "cache_committed_t": committed, "cache_commit_enqueued_t": committed,
                "cache_commit_timing": "host enqueue; final backend cache eval/sync included",
                "emitted_draft_count": min(accepted, len(new_tokens)),
                "emitted_target_count": max(0, len(new_tokens) - accepted),
            }
            recorder.commit(list(new_tokens), **fields)
            for token in new_tokens:
                yield int(token), None
                emitted += 1
                if emitted >= max_tokens:
                    return
            hidden = module._mtp_draft_hidden(model, verify.hidden[:, accepted:accepted + 1, :])
            bonus = int(new_tokens[-1])
            shared = module._slice_shared_kv_after_reject(verify.shared_kv_states, verified_count - accepted)
            offset += accepted + 1
            draft.set_shared_kv(shared, offset, position=module._mtp_draft_position(offset), kv_valid_len=offset)
            if emitted % 256 == 0:
                mx.clear_cache()

    def _measurement_counters(self, output_tokens):
        records = self._confidence_records
        position = 1
        for record in records:
            if record["output_count_before"] != position:
                raise RuntimeError("Confidence records do not cover the emitted sequence")
            position += len(record["emitted_token_ids"])
        if position != output_tokens:
            raise RuntimeError("Confidence records do not match generated token count")
        verified = [record for record in records if record["decision"] == "verify"]
        drafted = sum(record["verified_draft_count"] for record in verified)
        accepted = sum(record["accepted_count"] for record in verified)
        emitted_draft = sum(min(record["accepted_count"], len(record["emitted_token_ids"])) for record in verified)
        computed = sum(len(record["proposal_token_ids"]) for record in records)
        details = [{"drafted_tokens": record["verified_draft_count"],
                    "accepted_draft_tokens": record["accepted_count"],
                    "emitted_tokens": len(record["emitted_token_ids"]),
                    "emitted_draft_tokens": min(record["accepted_count"], len(record["emitted_token_ids"])),
                    "emitted_target_tokens": max(0, len(record["emitted_token_ids"]) - record["accepted_count"])}
                   for record in verified]
        return {
            "drafted_tokens": drafted, "accepted_draft_tokens": accepted,
            "draft_acceptance_rate": accepted / drafted if drafted else None,
            "speculative_rounds": len(verified),
            "mean_accepted_draft_tokens_per_round": accepted / len(verified) if verified else None,
            "emitted_draft_tokens": emitted_draft,
            "emitted_target_tokens": output_tokens - emitted_draft,
            "speculative_round_details": details,
            "computed_draft_tokens": computed, "verified_draft_tokens": drafted,
            "gated_out_draft_tokens": computed - drafted,
            "fallback_rounds": len(records) - len(verified),
            "confidence_rounds": records,
        }

    def metadata(self):
        metadata = super().metadata()
        metadata.update({
            "framework": "mlx-vlm-mtp-confidence",
            "algorithm": "native greedy MTP with additional confidence observation and optional post-draft gate",
            "confidence_policy": self.policy, "confidence_threshold": self.threshold,
            "sampling_contract": "strictly greedy target tokens only; stochastic sampling is unsupported",
            "round_timing_interpretation": "host-observed intervals with lazy GPU work; not isolated GPU kernel timings; measure confidence overhead using collect versus stock MTP",
            "round_timing_scope": "MTP head prefill before the first round, final GPU sync and Python logging remain included in decode but are not attributed to a round subphase",
            "confidence_definition": "exp(sum(logp)); each logp is float32 chosen-logit minus logsumexp over full vocabulary; no temperature or top-k filtering",
            "confidence_projection": "target.speculative_logits_from_hidden on normalized native drafter hidden states; original fused-argmax proposal IDs retained",
            "confidence_interpretation": "drafter sequence likelihood, not a calibrated probability of target acceptance",
            "confidence_overhead": "additional full vocabulary projection, scalar probability host read, event recording included in decode; no full-vocabulary CPU transfer",
            "gate_timing": "after all current proposals; draft cost is already spent; one-proposal budget tail uses p1 and p2=null",
            "fallback_cost": "one target verifier input plus native MTP cache/seed maintenance, including draft prefill; not equivalent in cost to MLXVLMBackend",
            "confidence_counters": "computed counts returned/scored proposals; verified counts only proposals offered to target; unused lookahead seed is maintenance, not a counted proposal; gated-out proposals are not target rejections",
            "acceptance_accounting": "only nonempty verified proposal blocks enter stock speculative acceptance metrics; unverified confidence accepted_count is null",
            "measurement_harness": "phase-separated native MTP with full prompt hidden capture and project confidence round loop",
        })
        return metadata
