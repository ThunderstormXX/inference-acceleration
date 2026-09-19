"""Record real, sequential autoregressive/MTP execution for an offline demo.

This is an instrumented demonstration, not a replacement for the benchmark.
Proposal IDs are read to the host before verification; cache commits retain the
stock asynchronous behavior. Display text is decoded after the timed request.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import gc
import importlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from inference_lab.core.config import ROOT
from inference_lab.core.io import environment, sha256_file, write_json
from inference_lab.backends.apple.mlx_memory import RecommendedWiredMemory


@contextmanager
def _patch_attribute(owner: Any, name: str, replacement: Any):
    """Restore both values and inherited-method lookup, even after failure."""
    had_local = name in vars(owner)
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if had_local:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


def build_prompt(tokenizer: Any, prompt: str, enable_thinking: bool = False) -> list[int]:
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        return_dict=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("Chat template must return a nonempty list of integer token IDs")
    return tokens


def _decode(tokenizer: Any, tokens: list[int]) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _common_prefix(left: str, right: str) -> str:
    end = 0
    for a, b in zip(left, right):
        if a != b:
            break
        end += 1
    return left[:end]


def display_tokens(tokenizer: Any, tokens: list[int]) -> dict[str, Any]:
    """Delay incomplete byte sequences until their stable Unicode text exists."""
    full = _decode(tokenizer, tokens)
    prefixes = []
    pieces = []
    previous = ""
    for index in range(1, len(tokens) + 1):
        candidate = _common_prefix(_decode(tokenizer, tokens[:index]), full)
        if len(candidate) < len(previous):
            candidate = previous
        pieces.append(candidate[len(previous):])
        prefixes.append(candidate)
        previous = candidate
    return {"output_text": full, "token_texts": pieces, "decoded_prefixes": prefixes}


def _enrich_events(result: dict[str, Any], tokenizer: Any) -> None:
    """Decode actual proposed IDs in their committed context, outside timing."""
    committed: list[int] = []
    for event in result["events"]:
        before = _decode(tokenizer, committed)
        after = _decode(tokenizer, committed + event["token_ids"])
        # UTF byte-fragment tokens may revise the final replacement character.
        # Record the exact contextual rendering and the stable insertion point.
        stable = _common_prefix(before, after)
        event["context_text"] = stable
        event["context_replaced_characters"] = len(before) - len(stable)
        event["text"] = after[len(stable):]
        prior = stable
        pieces = []
        for index in range(1, len(event["token_ids"]) + 1):
            prefix = _common_prefix(_decode(tokenizer, committed + event["token_ids"][:index]), after)
            if len(prefix) < len(prior):
                prefix = prior
            pieces.append(prefix[len(prior):])
            prior = prefix
        event["token_texts"] = pieces
        if event["type"] == "commit":
            committed.extend(event["token_ids"])
    if committed != result["token_ids"]:
        raise RuntimeError("Trace commit events do not reproduce the generated IDs")


class EventRecorder:
    def __init__(self, clock=perf_counter):
        self.clock = clock
        self.started = 0.0
        self.events: list[dict[str, Any]] = []
        self.token_ids: list[int] = []

    def start(self):
        self.started = self.clock()

    def now(self) -> float:
        return self.clock() - self.started

    def commit(self, tokens: list[int], **fields):
        self.token_ids.extend(tokens)
        self.events.append({"t": self.now(), "type": "commit", "token_ids": list(tokens),
                            "output_count": len(self.token_ids), **fields})

    def finish(self) -> dict[str, Any]:
        total = self.now()
        first = next(event["t"] for event in self.events if event["type"] == "commit")
        return {"token_ids": self.token_ids, "events": self.events,
                "prefill_seconds": first, "decode_seconds": total - first,
                "total_seconds": total}


class MTPTraceObserver:
    """Scoped observers around unchanged stock proposal/verification functions."""

    def __init__(self, backend: Any, recorder: EventRecorder, module: Any = None, *,
                 synchronize_first: bool = True, suppress_detokenization: bool = True,
                 record_first: bool = True):
        self.synchronize_first = synchronize_first
        self.suppress_detokenization = suppress_detokenization
        self.record_first = record_first
        self.backend = backend
        self.recorder = recorder
        self.module = module
        self.pending: dict[str, Any] | None = None
        self.round = 0
        self.yielded: list[int] = []

    def _evaluate(self, *values):
        if values:
            self.backend._mx.eval(*values)
        self.backend._mx.synchronize()

    @contextmanager
    def observe(self):
        module = self.module or importlib.import_module("mlx_vlm.speculative.mtp")
        draft = self.backend._draft
        original_draft = draft.draft_block
        original_verify = module._mtp_verify_target
        original_walk = module._mtp_acceptance_walk
        original_commit = module._MTPVerifyResult.commit
        original_rounds = self.backend._run_rounds

        def draft_block(*args, **kwargs):
            proposals = original_draft(*args, **kwargs)
            # This host read is needed to display actual proposals before their
            # verification. Do not force unrelated cache/hidden-state work.
            proposal_ids = [int(token) for token in proposals.reshape(-1).tolist()]
            self.round += 1
            self.pending = {"round": self.round, "proposed_token_ids": proposal_ids}
            self.recorder.events.append({"t": self.recorder.now(), "type": "draft",
                                         "token_ids": proposal_ids, "round": self.round,
                                         "output_count": len(self.recorder.token_ids)})
            return proposals

        def verify_target(*args, **kwargs):
            result = original_verify(*args, **kwargs)
            if self.pending is None or result.target_tokens is None:
                raise RuntimeError("Trace requires a pending draft and greedy target token verification")
            # The stock greedy acceptance walk already reads target IDs to the
            # host. Observe that boundary instead of inserting another barrier.
            self.pending["verification_result"] = result
            return result

        def acceptance_walk(*args, **kwargs):
            accepted, tokens = original_walk(*args, **kwargs)
            if self.pending is None:
                raise RuntimeError("MTP acceptance without a draft event")
            verification = self.pending.pop("verification_result")
            self.pending.update(
                verification_completed_t=self.recorder.now(),
                target_token_ids=[int(token) for token in verification.target_tokens.reshape(-1).tolist()],
                accepted_count=int(accepted), output_tokens=[int(t) for t in tokens],
            )
            return accepted, tokens

        def commit(verification, model, caches, accepted, block_size):
            value = original_commit(verification, model, caches, accepted, block_size)
            pending = self.pending
            if pending is None or pending.get("accepted_count") != accepted:
                raise RuntimeError("MTP commit does not match the observed acceptance decision")
            proposals = pending["proposed_token_ids"]
            if len(proposals) != block_size - 1 or not 0 <= accepted <= len(proposals):
                raise RuntimeError("Unexpected MTP block geometry")
            tokens = pending["output_tokens"]
            committed_t = self.recorder.now()
            self.recorder.commit(
                tokens, t=committed_t, round=pending["round"], accepted_count=accepted,
                draft_count=len(proposals), proposed_token_ids=proposals,
                rejected_token_ids=proposals[accepted:], target_token_ids=pending["target_token_ids"],
                verification_completed_t=pending["verification_completed_t"],
                cache_committed_t=committed_t, cache_commit_enqueued_t=committed_t,
                cache_commit_timing="host enqueue; GPU cache work may remain lazy until next round or final sync",
                emitted_draft_count=min(accepted, len(tokens)),
                emitted_target_count=max(0, len(tokens) - accepted),
            )
            self.pending = None
            return value

        def run_rounds(*args, **kwargs):
            stream = original_rounds(*args, **kwargs)
            try:
                for token, info in stream:
                    token = int(token)
                    first = not self.yielded
                    if first:
                        if self.synchronize_first:
                            self._evaluate()
                        if self.record_first:
                            self.recorder.commit([token], round=0, accepted_count=0, draft_count=0,
                                                 rejected_token_ids=[], emitted_draft_count=0, emitted_target_count=1)
                    position = len(self.yielded)
                    if not (first and not self.record_first):
                        if position >= len(self.recorder.token_ids) or self.recorder.token_ids[position] != token:
                            raise RuntimeError("MTP event output disagrees with stock generator output")
                    self.yielded.append(token)
                    yield token, info
            finally:
                stream.close()

        with ExitStack() as stack:
            stack.enter_context(_patch_attribute(draft, "draft_block", draft_block))
            stack.enter_context(_patch_attribute(module, "_mtp_verify_target", verify_target))
            stack.enter_context(_patch_attribute(module, "_mtp_acceptance_walk", acceptance_walk))
            stack.enter_context(_patch_attribute(module._MTPVerifyResult, "commit", commit))
            stack.enter_context(_patch_attribute(self.backend, "_run_rounds", run_rounds))
            # The backend normally decodes after its own phase clocks. Keep that
            # text work outside our whole-request clock as well.
            if self.suppress_detokenization:
                stack.enter_context(_patch_attribute(self.backend.tokenizer, "decode", lambda *a, **k: ""))
            yield self


def capture_baseline(backend: Any, prompt_tokens: list[int], count: int) -> dict[str, Any]:
    """The existing autoregressive loop, preserving its one-step GPU pipeline."""
    mx = backend._mx
    model = backend._model
    for attribute in ("_position_ids", "_rope_deltas"):
        if hasattr(model, attribute):
            setattr(model, attribute, None)
    recorder = EventRecorder()
    recorder.start()
    with RecommendedWiredMemory(mx):
        cache = backend._make_prompt_cache(model)
        try:
            prompt = mx.array(prompt_tokens, dtype=mx.uint32)
            mx.eval(prompt)
            mx.synchronize()
            mx.clear_cache()
            position = 0
            while position < len(prompt_tokens) - 1:
                end = min(position + backend.prefill_step_size, len(prompt_tokens) - 1)
                model(prompt[position:end][None], cache=cache)
                mx.eval([entry.state for entry in cache])
                mx.clear_cache()
                position = end
            token = backend._next_token(prompt[-1:], cache)
            mx.eval(token, [entry.state for entry in cache])
            mx.synchronize()
            recorder.commit([int(token.item())])
            for step in range(count - 1):
                following = backend._next_token(token, cache)
                mx.async_eval(following)
                if step:
                    recorder.commit([int(token.item())])
                token = following
                if (step + 1) % 256 == 0:
                    mx.clear_cache()
            if count > 1:
                recorder.commit([int(token.item())])
        finally:
            try:
                mx.eval([entry.state for entry in cache])
            finally:
                mx.synchronize()
    return recorder.finish()


def capture_mtp(backend: Any, prompt_tokens: list[int], count: int) -> dict[str, Any]:
    recorder = EventRecorder()
    observer = MTPTraceObserver(backend, recorder)
    with observer.observe():
        recorder.start()
        measurement = backend.measure(prompt_tokens, count)
        result = recorder.finish()
    if result["token_ids"] != measurement["generated_token_ids"] or observer.yielded != result["token_ids"]:
        raise RuntimeError("MTP trace and backend output disagree")
    result["acceptance"] = {key: measurement[key] for key in (
        "drafted_tokens", "accepted_draft_tokens", "draft_acceptance_rate", "speculative_rounds",
        "emitted_draft_tokens", "emitted_target_tokens",
    )}
    return result


def parity(baseline: list[int], speculative: list[int]) -> dict[str, Any]:
    mismatch = next((i for i, pair in enumerate(zip(baseline, speculative)) if pair[0] != pair[1]), None)
    if mismatch is None and len(baseline) != len(speculative):
        mismatch = min(len(baseline), len(speculative))
    return {"equal": mismatch is None, "compared_tokens": min(len(baseline), len(speculative)),
            "baseline_tokens": len(baseline), "mtp_tokens": len(speculative),
            "first_mismatch_index": mismatch,
            "baseline_token_at_mismatch": baseline[mismatch] if mismatch is not None and mismatch < len(baseline) else None,
            "mtp_token_at_mismatch": speculative[mismatch] if mismatch is not None and mismatch < len(speculative) else None}


@dataclass
class TraceConfig:
    prompt: str
    output: str
    model_path: str = str(ROOT / "models/qwen3.5-9b-mlx-4bit")
    draft_path: str = str(ROOT / "models/qwen3.5-9b-mtp-4bit")
    max_new_tokens: int = 192
    block_size: int = 3
    prefill_step_size: int = 512
    enable_thinking: bool = False

    def __post_init__(self):
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(self.block_size) is not int or not 2 <= self.block_size <= 5:
            raise ValueError("block_size must be between 2 and 5")
        if type(self.prefill_step_size) is not int or self.prefill_step_size < 1:
            raise ValueError("prefill_step_size must be a positive integer")


class TraceCapture:
    def __init__(self, config: TraceConfig):
        self.config = config

    def run(self) -> dict[str, Any]:
        lock_path = ROOT / "artifacts/gpu.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another inference benchmark holds the GPU lock") from exc
            result = self._run_pair()
            write_json(Path(self.config.output), result)
            return result

    def _run_pair(self) -> dict[str, Any]:
        from inference_lab.backends.apple.mlx_vlm_backend import MLXVLMBackend
        from inference_lab.backends.apple.speculative.mtp_backend import MTPBackend

        config = self.config
        result: dict[str, Any] = {
            "schema_version": 1, "prompt": config.prompt, "max_new_tokens": config.max_new_tokens,
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
                "environment": environment(), "warmup_requests_per_backend": 1,
                "run_order": ["baseline", "mtp"], "sampling": "greedy", "enable_thinking": config.enable_thinking,
                "ignore_eos": True, "fresh_cache_per_request": True,
                "instrumentation_policy": (
                    "Host perf_counter timestamps from each measured request start; load, tokenization, warmup "
                    "and display decoding excluded. Request setup, wired-memory scope and final synchronization included. "
                    "Baseline preserves one-step async pipeline and records host-observed token availability. "
                    "MTP reads draft proposal IDs before verification; verification completion is observed after the stock "
                    "acceptance walk has read target IDs. Commit timestamps mark native host commit enqueue, not GPU cache completion. "
                    "No per-round global synchronization or extra cache evaluation is inserted; final backend cache evaluation "
                    "and synchronization remain included. All observer overhead is counted. These demo times are not benchmark results."
                ),
                "trace_instrumentation_version": 2,
                "draft_rejection_policy": "rejected_token_ids is the discarded proposal suffix after the accepted prefix; it includes positions after the first mismatch",
                "display_policy": "special tokens hidden; incomplete Unicode byte fragments delayed until a stable prefix; proposal text decoded from actual proposal IDs in committed context",
                "timing_origin": "each backend has its own request start; the actual captures run sequentially",
            },
        }
        manifest = Path(config.model_path) / "download-manifest.json"
        if manifest.is_file():
            result["metadata"]["target_manifest_sha256"] = sha256_file(manifest)
            result["metadata"]["target_manifest"] = json.loads(manifest.read_text())
        prompt_ids = None
        for mode in ("baseline", "mtp"):
            backend = None
            runtime = None
            try:
                print(f"Loading {mode}", flush=True)
                if mode == "baseline":
                    backend = MLXVLMBackend(config.model_path, config.prefill_step_size, wired_memory=True)
                else:
                    backend = MTPBackend(config.model_path, config.draft_path, config.prefill_step_size,
                                         block_size=config.block_size, wired_memory=True)
                backend.load()
                runtime = backend._mx
                current_prompt = build_prompt(backend.tokenizer, config.prompt, config.enable_thinking)
                if prompt_ids is not None and current_prompt != prompt_ids:
                    raise RuntimeError("Baseline and MTP chat-template token IDs differ")
                prompt_ids = current_prompt
                result["prompt_token_ids"] = prompt_ids
                eos = getattr(backend.tokenizer, "eos_token_id", None)
                eos_ids = [eos] if type(eos) is int else list(eos or [])
                result["eos_token_ids"] = eos_ids
                print(f"Warmup {mode}: {config.max_new_tokens} tokens", flush=True)
                backend.measure(prompt_ids, config.max_new_tokens)
                print(f"Recording {mode}", flush=True)
                captured = (capture_baseline if mode == "baseline" else capture_mtp)(backend, prompt_ids, config.max_new_tokens)
                captured.update(display_tokens(backend.tokenizer, captured["token_ids"]))
                captured["eos_token_ids"] = eos_ids
                _enrich_events(captured, backend.tokenizer)
                result[mode] = captured
                result["metadata"][mode] = backend.metadata()
                print(f"Recorded {mode}: {len(captured['token_ids'])} tokens in {captured['total_seconds']:.3f}s", flush=True)
            finally:
                if backend is not None:
                    # Break the drafter's shared target binding as well as model
                    # references before the next large checkpoint is loaded.
                    if getattr(backend, "_draft", None) is not None:
                        backend._draft = None
                    backend._model = None
                    backend._tokenizer = None
                backend = None
                gc.collect()
                if runtime is not None:
                    runtime.synchronize()
                    runtime.clear_cache()
        result["parity"] = parity(result["baseline"]["token_ids"], result["mtp"]["token_ids"])
        return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/demos/mtp-race/trace.json")
    parser.add_argument("--model-path", default=TraceConfig.model_path)
    parser.add_argument("--draft-path", default=TraceConfig.draft_path)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args(argv)
    text = args.prompt if args.prompt is not None else args.prompt_file.read_text()
    config = TraceConfig(text.strip(), str(args.output.resolve()), args.model_path, args.draft_path,
                         args.max_new_tokens, args.block_size, args.prefill_step_size, args.enable_thinking)
    result = TraceCapture(config).run()
    print(f"Trace: {config.output}\nExact token parity: {result['parity']['equal']}", flush=True)
    return 0 if result["parity"]["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
