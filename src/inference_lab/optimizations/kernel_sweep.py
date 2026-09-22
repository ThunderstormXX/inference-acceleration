"""Bounded same-weight AR/DFlash kernel ablations on one resident target."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import traceback

from ..backends.apple.mlx_backend import MLXBackend
from ..backends.apple.speculative.dflash_backend import DFlashBackend
from ..profiling.dflash import plain_target
from ..profiling.runner import condition_snapshot, summarize
from ..benchmarking.metrics import validate_measurement
from ..core.activity import UserInitiatedActivity
from ..core.host_clock import MacSleepClock

ROOT = Path(__file__).resolve().parents[3]
# Values: (generation method, affine policy, fused argmax, ZMLX patterns)
VARIANTS = {
    "ar": ("ar", None, False, None),
    "dflash": ("dflash", None, False, None),
    "ar-head": ("ar", "head", False, None),
    "ar-target": ("ar", "target", False, None),
    "dflash-head": ("dflash", "head", False, None),
    "dflash-target": ("dflash", "target", False, None),
    "dflash-all": ("dflash", "all", False, None),
    "ar-argmax": ("ar", None, True, None),
    "dflash-argmax": ("dflash", None, True, None),
    "dflash-all-argmax": ("dflash", "all", True, None),
    "dflash-target-argmax": ("dflash", "target", True, None),
    "zmlx-deltanet": ("ar", None, False, ["deltanet"]),
    "zmlx-swiglu": ("ar", None, False, ["swiglu_mlp"]),
    "zmlx-both": ("ar", None, False, ["deltanet", "swiglu_mlp"]),
}


def summary(report):
    result = {"status": report.get("status", "running"), "by_variant": {}, "errors": len(report["errors"]),
              "all_runs_awake": all(not r["sleep"]["sleep_detected"] for r in report["runs"]) if report["runs"] else None,
              "all_token_ids_match": all(r["matches_baseline"] for r in report["runs"]) if report["runs"] else None}
    measured = [r for r in report["runs"] if r["phase"] == "measurement"]
    for variant in sorted({r["variant"] for r in measured}):
        rows = [r for r in measured if r["variant"] == variant]
        result["by_variant"][variant] = {
            "tok_s": summarize([r["tok_s"] for r in rows]),
            "prefill_ms": summarize([r["measurement"]["prefill_seconds"] * 1000 for r in rows]),
            "pooled_tok_s": sum(r["measurement"]["decode_tokens"] for r in rows) /
                            sum(r["measurement"]["decode_seconds"] for r in rows),
            "all_token_ids_match": all(r["matches_baseline"] for r in rows),
            "speedup_vs_matched_ar": summarize([
                next(b["measurement"]["decode_seconds"] for b in measured
                     if b["variant"] == "ar" and b["prompt_index"] == r["prompt_index"]
                     and b["repeat"] == r["repeat"]) / r["measurement"]["decode_seconds"]
                for r in rows if any(b["variant"] == "ar" and b["prompt_index"] == r["prompt_index"]
                                    and b["repeat"] == r["repeat"] for b in measured)])}
    return result


class KernelSweep:
    def __init__(self, args):
        self.args = args
        self.prompt_bytes = args.prompts.read_bytes()
        self.prompts = json.loads(self.prompt_bytes)[:args.prompt_count]
        if len(self.prompts) != args.prompt_count:
            raise ValueError("Not enough prompts")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.output = args.output or ROOT / "artifacts/optimizations" / (stamp + "-kernels")
        self.output.mkdir(parents=True, exist_ok=False)
        self.clock = MacSleepClock()
        self.report = {"schema_version": 1, "status": "running", "started": condition_snapshot(), "runs": [], "errors": [],
            "protocol": {"variants": args.variants, "tokens": args.tokens, "repeats": args.repeats,
                         "block_size": args.block_size, "prompts": self.prompts,
                         "prompts_sha256": sha256(self.prompt_bytes).hexdigest(),
                         "sampling": "greedy; EOS ignored; full vocabulary",
                         "trace_generation": args.trace,
                         "notes": ["First output token belongs to prefill; decode is N-1 tokens.",
                                   "Identical target weights; fresh cache per request; warm paths before timing.",
                                   "Variant order reverses on alternate repetitions; stock AR brackets entire sweep.",
                                   "DFlash draft remains resident during AR if any DFlash variant requested.",
                                   "Token parity is empirical for these prompts, not a universal mathematical guarantee."]}}
        self.baselines = {}
        self.failed = set()
        target = str(ROOT / "models/qwen3.5-9b-mlx-4bit")
        if any(VARIANTS[v][0] == "dflash" for v in args.variants):
            if args.trace:
                raise ValueError("DFlash token event tracing is not supported by this runner")
            self.backend = DFlashBackend(target, str(ROOT / "models/qwen3.5-9b-dflash"),
                                        block_size=args.block_size, draft_bits=4).load()
        else:
            self.backend = MLXBackend(target, wired_memory=True).load()
        self.backend.trace_generation = args.trace
        self.report["backend"] = self.backend.metadata()
        self.save()

    def save(self):
        temporary = self.output / "raw.json.gz.tmp"
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(self.report, stream, ensure_ascii=False)
        temporary.replace(self.output / "raw.json.gz")
        (self.output / "summary.json").write_text(json.dumps(summary(self.report), indent=2) + "\n")

    def measure(self, variant, prompt, tokens):
        generation, policy, fused, zmlx = VARIANTS[variant]
        contexts = []
        metadata = {}
        with ExitStack() as stack:
            if generation == "ar" and isinstance(self.backend, DFlashBackend):
                stack.enter_context(plain_target(self.backend))
            if policy:
                from .affine import ScopedAffineOptimization
                opt = ScopedAffineOptimization(self.backend._model, getattr(self.backend, "_draft", None), policy=policy)
                contexts.append(opt)
                stack.enter_context(opt)
            if fused:
                from .greedy_head import ScopedGreedyHead
                opt = ScopedGreedyHead(self.backend)
                contexts.append(opt)
                stack.enter_context(opt)
            if zmlx:
                from .zmlx import scoped_zmlx
                metadata["zmlx"] = stack.enter_context(scoped_zmlx(self.backend._model, zmlx))
            result = (MLXBackend.measure(self.backend, prompt, tokens) if generation == "ar"
                      else self.backend.measure(prompt, tokens))
        for opt in contexts:
            metadata[type(opt).__name__] = opt.metadata()
        return result, metadata

    def run_one(self, variant, pindex, repeat, phase):
        print(f"RUN {phase} {variant} p={pindex} repeat={repeat}", flush=True)
        start = perf_counter()
        clock_before = self.clock.snapshot()
        try:
            measurement, optimization = self.measure(variant, self.prompts[pindex]["prompt_tokens"], self.args.tokens)
            validate_measurement(measurement)
            ids = measurement["generated_token_ids"]
            if (len(ids) != self.args.tokens or measurement["generated_tokens"] != self.args.tokens
                    or any(type(t) is not int or t < 0 for t in ids)):
                raise ValueError("Invalid token IDs or fixed generation budget")
            if variant == "ar" and pindex not in self.baselines:
                self.baselines[pindex] = ids
            baseline = self.baselines[pindex]
            parity = ids == baseline
            row = {"variant": variant, "prompt_index": pindex, "repeat": repeat, "phase": phase,
                   "measurement": measurement, "optimization": optimization,
                   "wall_seconds": perf_counter() - start,
                   "sleep": MacSleepClock.assess(clock_before, self.clock.snapshot()),
                   "tok_s": measurement["decode_tokens"] / measurement["decode_seconds"],
                   "matches_baseline": parity}
            if not parity:
                row["first_mismatch"] = next((i for i, (a,b) in enumerate(zip(ids, baseline)) if a != b),
                                             min(len(ids), len(baseline)))
            self.report["runs"].append(row)
            print(f"DONE {variant}: {row['tok_s']:.3f} tok/s; parity={parity}", flush=True)
        except Exception:
            self.report["errors"].append({"variant": variant, "prompt_index": pindex,
                                          "phase": phase, "traceback": traceback.format_exc()})
            self.failed.add(variant)
            print(self.report["errors"][-1]["traceback"], flush=True)
            if variant == "ar":
                raise
        finally:
            self.save()

    def run(self):
        print(f"RESULTS {self.output}", flush=True)
        for variant in self.args.variants:
            try:
                self.measure(variant, self.prompts[0]["prompt_tokens"], 8)
            except Exception:
                self.failed.add(variant)
                self.report["errors"].append({"variant": variant, "phase": "warmup", "traceback": traceback.format_exc()})
                print(self.report["errors"][-1]["traceback"], flush=True)
                self.save()
                if variant == "ar":
                    raise
        self.report["after_warmup"] = condition_snapshot()
        # Reference runs precede reversed orders and are kept out of measured aggregates.
        for pindex in range(len(self.prompts)):
            self.run_one("ar", pindex, -1, "reference")
        for repeat in range(self.args.repeats):
            order = self.args.variants if repeat % 2 == 0 else list(reversed(self.args.variants))
            for pindex in range(len(self.prompts)):
                for variant in order:
                    if variant not in self.failed:
                        self.run_one(variant, pindex, repeat, "measurement")
        self.run_one("ar", 0, self.args.repeats, "bracket")
        self.report["finished"] = condition_snapshot()
        self.report["status"] = "completed_with_errors" if self.report["errors"] else "completed"
        self.save()
        print(json.dumps(summary(self.report), indent=2), flush=True)
        return 0 if (summary(self.report)["all_token_ids_match"] and summary(self.report)["all_runs_awake"]
                     and not self.report["errors"]) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["ar", "dflash", "ar-head", "dflash-head", "ar-target", "dflash-target", "dflash-all"])
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/profiling/dflash-prompts.json")
    parser.add_argument("--prompt-count", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--block-size", type=int, choices=range(2, 6), default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.prompt_count <= 5 or not 16 <= args.tokens <= 2048 or not 1 <= args.repeats <= 5:
        parser.error("Bounds: prompts1..5, tokens16..2048, repeats1..5")
    if len(set(args.variants)) != len(args.variants) or "ar" not in args.variants:
        parser.error("Require unique variants including stock ar")
    with (ROOT / "artifacts/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with UserInitiatedActivity(True, reason="User-requested bounded inference kernel sweep"):
            return KernelSweep(args).run()
