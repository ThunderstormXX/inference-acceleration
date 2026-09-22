"""Causal CPU suffix proposals with Qwen3.5's stock transactional verifier."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter

from ..mlx_vlm_backend import MLXVLMBackend
from ....optimizations.ngram import LongestSuffixProposer


def greedy_verified_prefix(proposal, target_ids):
    """Verifier position i predicts proposal i; the final slot is the bonus."""
    if (not proposal or len(target_ids) != len(proposal) + 1
            or any(type(t) is not int or t < 0 for t in (*proposal, *target_ids))):
        raise ValueError("Expected nonempty proposals and exactly one more target prediction")
    accepted = 0
    for proposed, target in zip(proposal, target_ids):
        if proposed != target:
            break
        accepted += 1
    return accepted, list(proposal[:accepted]) + [target_ids[accepted]]


def verify_proposal(model, cache, bonus, proposal, *, mx, verify_target, observer=None):
    """Commit bonus+accepted input tokens once; abort on any incomplete round.

    The newly emitted target correction/bonus is not yet in the target cache.
    ``commit`` accepts a draft count and internally retains accepted+1 inputs.
    The upstream helper owns abort if its forward itself raises before returning.
    """
    if not proposal:
        raise ValueError("A speculative verification requires a proposal")
    result = None
    try:
        inputs = mx.array([[bonus, *proposal]], dtype=mx.uint32)
        result = verify_target(model, inputs, cache, lambda logits: mx.argmax(logits, axis=-1),
                               sample_target_tokens=True)
        if result.target_tokens is None:
            raise RuntimeError("The greedy Qwen verifier did not return target token IDs")
        targets = result.target_tokens.reshape(-1).tolist()
        accepted, emitted = greedy_verified_prefix(proposal, targets)
        if observer is not None:
            observer.verified(targets)
        result.commit(model, cache, accepted, len(proposal) + 1)
        if observer is not None:
            observer.committed()
        return accepted, emitted
    except BaseException:
        if result is not None:
            result.abort()
        raise


class _NgramRoundTrace:
    """Observe existing host boundaries without adding GPU reads or fences."""

    def __init__(self, recorder, proposals, round_id):
        self.recorder = recorder
        self.proposals = list(proposals)
        self.round_id = round_id
        self.target_ids = None
        self.verification_completed_t = None
        self.cache_commit_enqueued_t = None
        recorder.events.append({"t": recorder.now(), "type": "draft",
                                "token_ids": list(proposals), "round": round_id,
                                "output_count": len(recorder.token_ids)})

    def verified(self, target_ids):
        self.target_ids = list(target_ids)
        self.verification_completed_t = self.recorder.now()

    def committed(self):
        self.cache_commit_enqueued_t = self.recorder.now()

    def finish(self, emitted, accepted):
        if self.verification_completed_t is None or self.cache_commit_enqueued_t is None:
            raise RuntimeError("Cannot record an incomplete retrieval verification")
        self.recorder.commit(
            emitted, round=self.round_id, ngram_round=self.round_id,
            accepted_count=accepted, draft_count=len(self.proposals),
            proposed_token_ids=list(self.proposals), rejected_token_ids=self.proposals[accepted:],
            target_token_ids=self.target_ids, verification_completed_t=self.verification_completed_t,
            cache_commit_enqueued_t=self.cache_commit_enqueued_t,
            cache_commit_timing="host enqueue; final cache evaluation/synchronization included in decode",
            emitted_draft_count=accepted, emitted_target_count=len(emitted) - accepted,
        )


class NgramBackend(MLXVLMBackend):
    """Use no neural draft, change no weights, and verify every copied token."""

    def __init__(self, model_path, prefill_step_size=512, *, width=2, min_match=2, max_match=16):
        super().__init__(model_path, prefill_step_size, wired_memory=True)
        if type(width) is not int or not 1 <= width <= 4:
            raise ValueError("width must be 1..4")
        LongestSuffixProposer(min_match=min_match, max_match=max_match)
        self.width, self.min_match, self.max_match = width, min_match, max_match
        self._verify_target = None
        self._generation_stream = None
        self._source_hashes = {}

    def load(self):
        if self._verify_target is not None:
            return self
        super().load()
        from mlx_vlm.speculative import mtp, cache_state, common
        from mlx_vlm.models.qwen3_5 import speculative_verifier
        if not callable(getattr(self._model, "speculative_verify_hidden", None)):
            raise ValueError("Ngram requires the Qwen3.5 transactional speculative verifier")
        self._verify_target = mtp._mtp_verify_target
        self._generation_stream = common.generation_stream
        for module in (mtp, cache_state, speculative_verifier):
            path = Path(module.__file__)
            self._source_hashes[module.__name__] = sha256(path.read_bytes()).hexdigest()
        return self

    def warm_verifier(self, prompt_tokens, width):
        """Compile a discarded transaction of each width even without a suffix hit."""
        if type(width) is not int or not 1 <= width <= 4:
            raise ValueError("warmup width must be 1..4")
        from ..mlx_memory import RecommendedWiredMemory
        for attribute in ("_position_ids", "_rope_deltas"):
            if hasattr(self._model, attribute):
                setattr(self._model, attribute, None)
        cache = self._make_prompt_cache(self._model)
        mx = self._mx
        with RecommendedWiredMemory(mx), mx.stream(self._generation_stream):
            token = self._next_token(mx.array(prompt_tokens, dtype=mx.uint32), cache)
            mx.eval(token, [entry.state for entry in cache])
            bonus = int(token.item())
            verify_proposal(self._model, cache, bonus, [bonus] * width,
                            mx=mx, verify_target=self._verify_target)
            mx.eval([entry.state for entry in cache])
            mx.synchronize()

    def _measure(self, prompt_tokens, max_new_tokens=128):
        if self._model is None or self._verify_target is None:
            raise RuntimeError("Call load() before measure()")
        if (not prompt_tokens or any(type(t) is not int or t < 0 for t in prompt_tokens)
                or type(max_new_tokens) is not int or max_new_tokens < 1):
            raise ValueError("Expected nonempty integer prompt IDs and a positive output budget")
        proposer = LongestSuffixProposer(min_match=self.min_match, max_match=self.max_match)
        mx = self._mx
        cache = self._make_prompt_cache(self._model)
        prompt = mx.array(prompt_tokens, dtype=mx.uint32)
        mx.eval(prompt)
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        recorder = None
        if self.trace_generation:
            from ....visualization.recording import GenerationRecorder
            recorder = GenerationRecorder(self, clock=perf_counter, speculative=True)
        started = perf_counter()
        if recorder is not None:
            recorder.start_at(started)
        position = 0
        while position < len(prompt_tokens) - 1:
            end = min(position + self.prefill_step_size, len(prompt_tokens) - 1)
            self._model(prompt[position:end][None], cache=cache)
            mx.eval([entry.state for entry in cache])
            mx.clear_cache()
            position = end
        token = self._next_token(prompt[-1:], cache)
        mx.eval(token, [entry.state for entry in cache])
        mx.synchronize()
        first = int(token.item())
        generated = [first]
        proposer.commit(prompt_tokens)
        proposer.commit([first])
        if recorder is not None:
            recorder.commit([first], round=0, accepted_count=0, draft_count=0,
                            rejected_token_ids=[], emitted_draft_count=0, emitted_target_count=1)
        prefill_seconds = perf_counter() - started
        decode_started = started + prefill_seconds if recorder is not None else perf_counter()
        rounds = []
        proposed_total = accepted_total = fallback_rounds = 0
        lookup_seconds = 0.0
        try:
            while len(generated) < max_new_tokens:
                round_started = perf_counter()
                remaining = max_new_tokens - len(generated)
                lookup_start = perf_counter()
                proposal = proposer.propose(min(self.width, remaining - 1))
                lookup_seconds += perf_counter() - lookup_start
                drafted = list(proposal.token_ids)
                round_trace = (_NgramRoundTrace(recorder, drafted, len(rounds) + 1)
                               if recorder is not None and drafted else None)
                with mx.stream(self._generation_stream):
                    if drafted:
                        accepted, emitted = verify_proposal(
                            self._model, cache, generated[-1], drafted, mx=mx, verify_target=self._verify_target,
                            observer=round_trace)
                    else:
                        next_token = self._next_token(mx.array([generated[-1]], dtype=mx.uint32), cache)
                        mx.eval(next_token)
                        emitted, accepted = [int(next_token.item())], 0
                        fallback_rounds += 1
                generated.extend(emitted)
                # Only actual verified output enters the retrieval history.
                proposer.commit(emitted)
                proposed_total += len(drafted)
                accepted_total += accepted
                rounds.append({"round": len(rounds) + 1, "proposal_token_ids": drafted,
                               "accepted_draft_tokens": accepted, "emitted_token_ids": emitted,
                               "matched_suffix_length": proposal.matched_suffix_length,
                               "source_start": proposal.source_start, "source_end": proposal.source_end,
                               "available_history_tokens_before": proposer.committed_tokens - len(emitted),
                               "start_decode_seconds": round_started - decode_started,
                               "end_decode_seconds": perf_counter() - decode_started})
                if round_trace is not None:
                    round_trace.finish(emitted, accepted)
                elif recorder is not None:
                    # Empty proposals are ordinary AR events, not speculative
                    # rounds. Keep the retrieval loop index in a separate field.
                    recorder.commit(emitted, round=0, ngram_round=len(rounds),
                                    accepted_count=0, draft_count=0, proposed_token_ids=[],
                                    rejected_token_ids=[], emitted_draft_count=0,
                                    emitted_target_count=1)
                if len(rounds) % 256 == 0:
                    mx.clear_cache()
        finally:
            mx.eval([entry.state for entry in cache])
            mx.synchronize()
        decode_seconds = perf_counter() - decode_started if max_new_tokens > 1 else 0.0
        result = {"prompt_tokens": len(prompt_tokens), "generated_tokens": len(generated),
                  "decode_tokens": len(generated) - 1, "prefill_seconds": prefill_seconds,
                  "decode_seconds": decode_seconds, "generated_token_ids": generated,
                  "output_text": self.tokenizer.decode(generated, skip_special_tokens=False),
                  "peak_memory_gb": float(mx.get_peak_memory()) / 1_000_000_000,
                  "drafted_tokens": proposed_total, "accepted_draft_tokens": accepted_total,
                  "draft_acceptance_rate": accepted_total / proposed_total if proposed_total else None,
                  "speculative_rounds": len(rounds) - fallback_rounds, "fallback_rounds": fallback_rounds,
                  "total_rounds": len(rounds), "round_details": rounds,
                  "lookup_seconds": lookup_seconds,
                  "lookup_timing_note": "CPU propose only; index updates and all orchestration are included in total decode.",
                  "timing_method": "Synchronized phase wall time; stock transactional verification; final cache materialization included; per-round times are host observations."}
        if recorder is not None:
            result["generation_trace"] = recorder.finish_measurement(result)
            from ....visualization.recording import validate_generation_trace
            validate_generation_trace(result)
        return result

    def metadata(self):
        result = super().metadata()
        result.update({"framework": "mlx-vlm-ngram", "algorithm": "causal longest-suffix retrieval + stock Qwen3.5 greedy verification",
                       "width": self.width, "min_match": self.min_match, "max_match": self.max_match,
                       "neural_draft": False, "retrieval_corpus": "current prompt + committed generated tokens only",
                       "upstream_source_sha256": self._source_hashes})
        return result
