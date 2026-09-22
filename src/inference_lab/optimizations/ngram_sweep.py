"""Bounded live retrieval-versus-AR pairs using one resident Qwen target."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from ..backends.apple.speculative.ngram_backend import NgramBackend
from ..benchmarking.metrics import validate_measurement
from ..core.activity import UserInitiatedActivity
from ..core.config import ROOT
from ..core.host_clock import MacSleepClock
from ..core.io import environment, power_thermal_snapshot, write_json
from .mtp_sweep import SharedTargetBaseline, load_prompts, token_comparison, _stats


def summarize(report):
    rows = report["runs"]
    result = {"status": report["status"], "completed_runs": len(rows),
              "all_token_ids_match": all(r["matches_baseline"] for r in rows) if rows else None,
              "all_runs_awake": all(not r["sleep"]["sleep_detected"] for r in rows) if rows else None,
              "by_variant": {}, "pairs": []}
    measured = [r for r in rows if r["role"] == "paired"]
    for variant in dict.fromkeys(r["variant"] for r in measured):
        selected = [r["measurement"] for r in measured if r["variant"] == variant]
        result["by_variant"][variant] = {
            "tok_s": _stats([m["decode_tokens"] / m["decode_seconds"] for m in selected]),
            "pooled_tok_s": sum(m["decode_tokens"] for m in selected) / sum(m["decode_seconds"] for m in selected),
            "prefill_ms": _stats([1000 * m["prefill_seconds"] for m in selected])}
        if variant != "ar":
            drafted = sum(m["drafted_tokens"] for m in selected)
            result["by_variant"][variant].update({
                "drafted_tokens": drafted, "accepted_draft_tokens": sum(m["accepted_draft_tokens"] for m in selected),
                "acceptance": sum(m["accepted_draft_tokens"] for m in selected) / drafted if drafted else None,
                "lookup_rounds": sum(m["speculative_rounds"] for m in selected),
                "fallback_rounds": sum(m["fallback_rounds"] for m in selected),
                "useful_tokens_per_round": sum(m["decode_tokens"] for m in selected) / sum(m["total_rounds"] for m in selected)})
    for pair_id in dict.fromkeys(r["pair_id"] for r in measured):
        pair = [r for r in measured if r["pair_id"] == pair_id]
        if len(pair) == 2:
            ar = next(r for r in pair if r["variant"] == "ar")
            method = next(r for r in pair if r["variant"] != "ar")
            result["pairs"].append({"pair_id": pair_id, "variant": method["variant"],
                "decode_speed_ratio": ar["measurement"]["decode_seconds"] / method["measurement"]["decode_seconds"],
                "token_ids_match": ar["measurement"]["generated_token_ids"] == method["measurement"]["generated_token_ids"]})
    for variant, result_row in result["by_variant"].items():
        if variant != "ar":
            result_row["paired_speed_ratio"] = _stats([p["decode_speed_ratio"] for p in result["pairs"] if p["variant"] == variant])
    brackets = [r["measurement"] for r in rows if r["role"] == "bracket"]
    result["drift_bracket"] = {"before_tok_s": brackets[0]["decode_tokens"] / brackets[0]["decode_seconds"],
        "after_tok_s": brackets[-1]["decode_tokens"] / brackets[-1]["decode_seconds"],
        "after_over_before": brackets[0]["decode_seconds"] / brackets[-1]["decode_seconds"]} if len(brackets) == 2 else None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/profiling/dflash-prompts.json")
    parser.add_argument("--prompt-count", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--min-matches", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.prompt_count <= 5 or not 16 <= args.tokens <= 512 or not 1 <= args.repeats <= 3:
        parser.error("Bounds: 1..5 prompts, 16..512 tokens, 1..3 repeats")
    if (not args.widths or any(k not in (1, 2, 4) for k in args.widths) or len(set(args.widths)) != len(args.widths)
            or not args.min_matches or any(n not in (2, 4, 8) for n in args.min_matches)
            or len(set(args.min_matches)) != len(args.min_matches)):
        parser.error("Unique widths from 1,2,4 and minimum matches from 2,4,8 required")
    raw, prompts = load_prompts(args.prompts, args.prompt_count)
    variants = [(width, minimum) for minimum in args.min_matches for width in args.widths]
    output = args.output or ROOT / "artifacts/optimizations" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-ngram")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "running", "environment": environment(), "runs": [],
              "started_conditions": power_thermal_snapshot(),
              "protocol": {"tokens": args.tokens, "repeats": args.repeats, "widths": args.widths,
                           "min_matches": args.min_matches, "prompts": prompts,
                           "prompts_sha256": sha256(raw).hexdigest(), "trace": args.trace,
                           "notes": ["Fixed-budget greedy; first token in prefill; EOS ignored.",
                                     "Same loaded target and no neural draft; fresh request caches.",
                                     "Longest suffix from current prompt and committed output only; no teacher answers.",
                                     "Transactional stock MTP verifier commits accepted+1 inputs; no direct hybrid-cache trimming.",
                                     "Each retrieval configuration has a neighboring AR; reverse pair order on repeat 2.",
                                     "Discarded explicit verifier warmups cover every width before measured generation.",
                                     "Per-round times are host observations; no extra per-operation GPU synchronization."]}}
    def save():
        write_json(output / "raw.json", report)
        write_json(output / "summary.json", summarize(report))
    clock = MacSleepClock()
    baselines = {}
    save()
    try:
        with (ROOT / "artifacts/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with UserInitiatedActivity(True, reason="User-requested bounded causal retrieval decoding experiment"):
                backend = NgramBackend(str(ROOT / "models/qwen3.5-9b-mlx-4bit")).load()
                baseline = SharedTargetBaseline(backend)
                report["backend"] = backend.metadata()
                print(f"Loaded target without neural draft; results: {output}", flush=True)
                baseline.measure(prompts[0]["prompt_tokens"], 16)
                for width, minimum in variants:
                    backend.width, backend.min_match = width, minimum
                    backend.measure(prompts[0]["prompt_tokens"], 16)
                for width in range(1, max(args.widths) + 1):
                    backend.warm_verifier(prompts[0]["prompt_tokens"], width)
                backend.trace_generation = baseline.trace_generation = args.trace
                report["after_warmup"] = power_thermal_snapshot()

                def run(mode, pindex, repeat, *, width=0, minimum=0, role="paired", pair_id=None):
                    variant = "ar" if mode == "ar" else f"ngram-w{width}-m{minimum}"
                    print(f"RUN {role} {variant} prompt={pindex} repeat={repeat}", flush=True)
                    before = clock.snapshot()
                    started = perf_counter()
                    if mode == "ar":
                        m = baseline.measure(prompts[pindex]["prompt_tokens"], args.tokens)
                    else:
                        backend.width, backend.min_match = width, minimum
                        m = backend.measure(prompts[pindex]["prompt_tokens"], args.tokens)
                    elapsed = perf_counter() - started
                    sleep = MacSleepClock.assess(before, clock.snapshot())
                    validate_measurement(m)
                    ids = m["generated_token_ids"]
                    if (len(ids) != args.tokens or m["generated_tokens"] != args.tokens
                            or any(type(t) is not int or t < 0 for t in ids)):
                        raise RuntimeError("Invalid generated token IDs or output budget")
                    if mode == "ar":
                        baselines.setdefault(pindex, ids)
                    row = {"variant": variant, "width": width, "min_match": minimum,
                           "prompt_index": pindex, "repeat": repeat, "role": role, "pair_id": pair_id,
                           "measurement": m, "sleep": sleep, "wall_seconds": elapsed,
                           **token_comparison(ids, baselines[pindex])}
                    report["runs"].append(row)
                    save()
                    print(f"DONE {variant}: {m['decode_tokens'] / m['decode_seconds']:.3f} tok/s; parity={row['matches_baseline']}", flush=True)
                    if sleep["sleep_detected"]:
                        raise RuntimeError("Host slept during measurement")

                run("ar", 0, -1, role="bracket")
                for pindex in range(1, len(prompts)):
                    run("ar", pindex, -1, role="reference")
                for repeat in range(args.repeats):
                    selected = variants if repeat % 2 == 0 else list(reversed(variants))
                    for pindex in range(len(prompts)):
                        for width, minimum in selected:
                            order = ("ar", "ngram") if (repeat + pindex + variants.index((width, minimum))) % 2 == 0 else ("ngram", "ar")
                            pair_id = f"r{repeat}-p{pindex}-w{width}-m{minimum}"
                            for mode in order:
                                run(mode, pindex, repeat, width=width, minimum=minimum, pair_id=pair_id)
                            report["runs"][-1]["conditions_after_pair"] = power_thermal_snapshot()
                            save()
                run("ar", 0, args.repeats, role="bracket")
        report["status"] = "completed" if summarize(report)["all_token_ids_match"] else "parity_failed"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["finished_conditions"] = power_thermal_snapshot()
        save()
    print(json.dumps(summarize(report), indent=2), flush=True)
    return 0 if report["status"] == "completed" else 2
