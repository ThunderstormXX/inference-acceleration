"""Paired native-MTP depth sweep on one resident target and draft checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import statistics
from time import perf_counter

from ..backends.apple.mlx_vlm_backend import MLXVLMBackend
from ..backends.apple.speculative.mtp_backend import MTPBackend
from ..benchmarking.metrics import validate_measurement
from ..core.activity import UserInitiatedActivity
from ..core.config import ROOT
from ..core.host_clock import MacSleepClock
from ..core.io import environment, power_thermal_snapshot, write_json


@dataclass(frozen=True)
class SweepConfig:
    prompt_count: int = 3
    tokens: int = 128
    repeats: int = 2
    block_sizes: tuple[int, ...] = (2, 3, 4, 5)
    trace: bool = False

    def __post_init__(self):
        for name, low, high in (("prompt_count", 1, 5), ("tokens", 16, 512), ("repeats", 1, 5)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
        if (not self.block_sizes or len(set(self.block_sizes)) != len(self.block_sizes)
                or any(type(k) is not int or k not in (2, 3, 4, 5) for k in self.block_sizes)):
            raise ValueError("block_sizes must contain unique integers in 2..5")
        if type(self.trace) is not bool:
            raise ValueError("trace must be boolean")


def paired_plan(config: SweepConfig) -> list[dict]:
    """Reverse depths and each depth's pair order on the second repetition."""
    result = []
    for repeat in range(config.repeats):
        depths = config.block_sizes if repeat % 2 == 0 else tuple(reversed(config.block_sizes))
        for prompt_index in range(config.prompt_count):
            for k in depths:
                depth_index = config.block_sizes.index(k)
                modes = ("ar", "mtp") if (repeat + prompt_index + depth_index) % 2 == 0 else ("mtp", "ar")
                pair_id = f"r{repeat}-p{prompt_index}-k{k}"
                for mode in modes:
                    result.append({"mode": mode, "block_size": k if mode == "mtp" else 1,
                                   "paired_block_size": k, "prompt_index": prompt_index,
                                   "repeat": repeat, "pair_id": pair_id, "role": "paired"})
    return result


def stock_comparison_plan(config: SweepConfig) -> list[dict]:
    """Add paired stock-MTP controls, reversing control/treatment block order."""
    plan = paired_plan(config)
    result = []
    for index in range(0, len(plan), 2):
        treatment = plan[index:index + 2]
        control = [{**row, "mode": "mtp-stock" if row["mode"] == "mtp" else "ar",
                    "pair_id": row["pair_id"] + "-stock"} for row in treatment]
        blocks = [control, treatment] if treatment[0]["repeat"] % 2 == 0 else [treatment, control]
        for block in blocks:
            result.extend(block)
    return result


class SharedTargetBaseline(MLXVLMBackend):
    """An AR adapter sharing weights, but never MTP's overridden _measure.

    Calling MLXVLMBackend.measure(mtp_backend, ...) would still dispatch to
    mtp_backend._measure. A distinct adapter retains AR dispatch and the same
    rotary reset, wired-memory policy, target object, tokenizer and cache factory.
    """

    def __init__(self, mtp: MTPBackend):
        super().__init__(mtp.model_path, mtp.prefill_step_size, wired_memory=True)
        if mtp._model is None:
            raise ValueError("The shared target must already be loaded")
        for name in ("_mx", "_model", "_tokenizer", "_make_prompt_cache", "_model_config"):
            setattr(self, name, getattr(mtp, name))


def token_comparison(actual: list[int], expected: list[int]) -> dict:
    first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
    if first is None and len(actual) != len(expected):
        first = min(len(actual), len(expected))
    return {"matches_baseline": actual == expected, "first_mismatch_index": first}


def refresh_parity(report: dict) -> None:
    baselines = {}
    for row in report["runs"]:
        if row["mode"] == "ar":
            baselines.setdefault(row["prompt_index"], row["measurement"]["generated_token_ids"])
    for row in report["runs"]:
        expected = baselines.get(row["prompt_index"])
        row.update(token_comparison(row["measurement"]["generated_token_ids"], expected)
                   if expected is not None else {"matches_baseline": None, "first_mismatch_index": None})


def _stats(values: list[float]):
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)} if values else None


def compact_summary(report: dict) -> dict:
    refresh_parity(report)
    rows = report["runs"]
    checked = [r for r in rows if r.get("matches_baseline") is not None]
    result = {"status": report["status"], "completed_runs": len(rows),
              "expected_runs": report["protocol"]["expected_runs"],
              "all_token_ids_match": all(r["matches_baseline"] for r in checked) if checked and len(checked) == len(rows) else None,
              "all_runs_awake": all(not r["sleep"]["sleep_detected"] for r in rows) if rows else None,
              "by_mode": {}, "pairs": [], "drift_bracket": None}
    paired = [r for r in rows if r["role"] == "paired"]
    for mode, k in sorted({(r["mode"], r["block_size"]) for r in paired}):
        selected = [r["measurement"] for r in paired if (r["mode"], r["block_size"]) == (mode, k)]
        result["by_mode"][f"{mode}_k{k}"] = {
            "tok_s": _stats([m["decode_tokens"] / m["decode_seconds"] for m in selected]),
            "pooled_tok_s": sum(m["decode_tokens"] for m in selected) / sum(m["decode_seconds"] for m in selected),
            "ms_per_token": _stats([1000 * m["decode_seconds"] / m["decode_tokens"] for m in selected]),
            "prefill_ms": _stats([1000 * m["prefill_seconds"] for m in selected])}
        if mode != "ar":
            result["by_mode"][f"{mode}_k{k}"].update({
                "draft_acceptance": _stats([m["draft_acceptance_rate"] for m in selected]),
                "useful_tokens_per_round": _stats([m["decode_tokens"] / m["speculative_rounds"] for m in selected])})
    for pair_id in dict.fromkeys(r["pair_id"] for r in paired):
        pair = [r for r in paired if r["pair_id"] == pair_id]
        if len(pair) != 2:
            continue
        ar = next(r for r in pair if r["mode"] == "ar")
        mtp = next(r for r in pair if r["mode"] != "ar")
        result["pairs"].append({"pair_id": pair_id, "method": mtp["mode"], "block_size": mtp["block_size"],
            "prompt_index": mtp["prompt_index"], "repeat": mtp["repeat"], "order": [r["mode"] for r in pair],
            "decode_speed_ratio": ar["measurement"]["decode_seconds"] / mtp["measurement"]["decode_seconds"],
            "pair_token_ids_match": ar["measurement"]["generated_token_ids"] == mtp["measurement"]["generated_token_ids"]})
    for mode, k in sorted({(r["mode"], r["block_size"]) for r in paired if r["mode"] != "ar"}):
        result["by_mode"][f"{mode}_k{k}"]["paired_speed_ratio"] = _stats([
            p["decode_speed_ratio"] for p in result["pairs"] if p["block_size"] == k and p["method"] == mode])
    comparisons = []
    for row in paired:
        if row["mode"] != "mtp-stock":
            continue
        treatment = next((r for r in paired if r["mode"] == "mtp" and all(
            r[key] == row[key] for key in ("prompt_index", "repeat", "block_size"))), None)
        if treatment:
            comparisons.append({"prompt_index": row["prompt_index"], "repeat": row["repeat"],
                "block_size": row["block_size"],
                "shortlist_over_stock": row["measurement"]["decode_seconds"] / treatment["measurement"]["decode_seconds"],
                "token_ids_match": row["measurement"]["generated_token_ids"] == treatment["measurement"]["generated_token_ids"]})
    result["draft_comparisons"] = comparisons
    result["shortlist_over_stock"] = _stats([row["shortlist_over_stock"] for row in comparisons])
    bracket = [r for r in rows if r["role"] == "bracket"]
    if len(bracket) == 2:
        before, after = (r["measurement"] for r in bracket)
        result["drift_bracket"] = {"prompt_index": 0,
            "before_tok_s": before["decode_tokens"] / before["decode_seconds"],
            "after_tok_s": after["decode_tokens"] / after["decode_seconds"],
            "after_over_before": before["decode_seconds"] / after["decode_seconds"]}
    return result


def load_prompts(path: Path, count: int) -> tuple[bytes, list[dict]]:
    raw = path.read_bytes()
    prompts = json.loads(raw)
    if not isinstance(prompts, list) or len(prompts) < count:
        raise ValueError("The prompt file has fewer rows than requested")
    prompts = prompts[:count]
    for row in prompts:
        ids = row.get("prompt_tokens") if isinstance(row, dict) else None
        if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError("Each prompt must contain a nonempty list of nonnegative integer IDs")
    return raw, prompts


class MTPSweepRunner:
    def __init__(self, config: SweepConfig, prompts_path: Path, output: Path,
                 model_path: Path, draft_path: Path, *, optimization_metadata=None, draft_vocab_path=None, compare_stock_draft=False):
        self.config, self.prompts_path, self.output = config, prompts_path, output
        self.model_path, self.draft_path = model_path, draft_path
        self.optimization_metadata = optimization_metadata
        self.draft_vocab_path = draft_vocab_path
        self.compare_stock_draft = compare_stock_draft
        if compare_stock_draft and draft_vocab_path is None:
            raise ValueError("Stock-draft comparison requires a vocabulary")

    def run(self) -> int:
        config = self.config
        raw, prompts = load_prompts(self.prompts_path, config.prompt_count)
        plan = stock_comparison_plan(config) if self.compare_stock_draft else paired_plan(config)
        self.output.mkdir(parents=True, exist_ok=False)
        report = {"schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                  "environment": environment(), "started_conditions": power_thermal_snapshot(), "runs": [],
                  "target_manifest_sha256": sha256((self.model_path / "download-manifest.json").read_bytes()).hexdigest(),
                  "optimization": self.optimization_metadata,
                  "protocol": {"prompts_source": str(self.prompts_path.resolve()), "prompts_sha256": sha256(raw).hexdigest(),
                      "prompts": prompts, "tokens": config.tokens, "repeats": config.repeats,
                      "block_sizes": list(config.block_sizes), "trace": config.trace, "expected_runs": len(plan) + 2,
                      "compare_stock_draft": self.compare_stock_draft,
                      "notes": ["Fixed-budget greedy; EOS ignored; first token in prefill, N-1 decode tokens.",
                                "One target object; draft weights remain resident for AR; fresh request caches.",
                                "Every K has a neighboring matched AR; pair order and depth order reverse across repetitions.",
                                "All methods warmed for 16 tokens; start/end AR drift bracket excluded from averages.",
                                "Timings include final cache materialization; text decoding and host snapshots are outside decode.",
                                "Stdev describes this small repeated-prompt diagnostic, not independent task-population uncertainty."]}}

        draft_vocab_opt = None

        def save():
            if draft_vocab_opt is not None:
                report["draft_vocabulary"] = draft_vocab_opt.metadata()
            summary = compact_summary(report)
            write_json(self.output / "raw.json", report)
            write_json(self.output / "summary.json", summary)

        activity = UserInitiatedActivity(True, reason="User-requested bounded native MTP depth sweep")
        clock = MacSleepClock()
        save()
        try:
            with (ROOT / "artifacts/gpu.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("Another inference benchmark holds the GPU lock") from exc
                with activity, ExitStack() as optimization_scopes:
                    backend = MTPBackend(str(self.model_path), str(self.draft_path), block_size=config.block_sizes[0]).load()
                    if self.draft_vocab_path is not None:
                        from .draft_vocab import ScopedDraftVocabulary
                        draft_vocab_opt = ScopedDraftVocabulary(backend, self.draft_vocab_path)
                        optimization_scopes.enter_context(draft_vocab_opt)
                    baseline = SharedTargetBaseline(backend)
                    report["backend"] = backend.metadata()
                    report["baseline"] = baseline.metadata()
                    print(f"Loaded shared target and MTP draft; results: {self.output}", flush=True)
                    baseline.measure(prompts[0]["prompt_tokens"], 16)
                    for k in config.block_sizes:
                        backend.block_size = k
                        backend.measure(prompts[0]["prompt_tokens"], 16)
                        if self.compare_stock_draft:
                            with draft_vocab_opt.suspended():
                                backend.measure(prompts[0]["prompt_tokens"], 16)
                    baseline.trace_generation = backend.trace_generation = config.trace
                    report["after_warmup"] = power_thermal_snapshot()
                    save()

                    def run_one(spec):
                        before = clock.snapshot()
                        started = perf_counter()
                        print(f"RUN {spec['mode']} K={spec['block_size']} prompt={spec['prompt_index']} repeat={spec['repeat']}", flush=True)
                        adapter = baseline if spec["mode"] == "ar" else backend
                        if spec["mode"] != "ar":
                            backend.block_size = spec["block_size"]
                        with draft_vocab_opt.suspended() if spec["mode"] == "mtp-stock" else nullcontext():
                            measurement = adapter.measure(prompts[spec["prompt_index"]]["prompt_tokens"], config.tokens)
                        elapsed = perf_counter() - started
                        sleep = MacSleepClock.assess(before, clock.snapshot())
                        validate_measurement(measurement)
                        ids = measurement["generated_token_ids"]
                        if (len(ids) != config.tokens or measurement["generated_tokens"] != config.tokens
                                or any(type(t) is not int or t < 0 for t in ids)):
                            raise RuntimeError("Measured output does not satisfy the fixed token budget")
                        report["runs"].append({**spec, "measurement": measurement, "wall_seconds": elapsed, "sleep": sleep})
                        save()
                        print(f"DONE {measurement['decode_tokens'] / measurement['decode_seconds']:.3f} tok/s; parity={report['runs'][-1]['matches_baseline']}", flush=True)
                        if sleep["sleep_detected"]:
                            raise RuntimeError("Host slept during measurement; saved evidence is not a valid speed benchmark")

                    bracket = {"mode": "ar", "block_size": 1, "paired_block_size": None, "prompt_index": 0,
                               "repeat": -1, "pair_id": None, "role": "bracket"}
                    run_one(bracket)
                    for number, spec in enumerate(plan):
                        run_one(spec)
                        if number % 2:
                            report["runs"][-1]["conditions_after_pair"] = power_thermal_snapshot()
                            save()
                    run_one({**bracket, "repeat": config.repeats})
            summary = compact_summary(report)
            report["status"] = "completed" if summary["all_token_ids_match"] else "parity_failed"
        except BaseException as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            report["finished_conditions"] = power_thermal_snapshot()
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            report["activity"] = activity.metadata()
            save()
        print(f"COMPLETE {self.output}; status={report['status']}", flush=True)
        return 0 if report["status"] == "completed" else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/profiling/dflash-prompts.json")
    parser.add_argument("--prompt-count", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--compare-stock-draft", action="store_true", help="Also interleave stock MTP with the same resident shortlist arrays")
    parser.add_argument("--draft-vocab", type=Path, help="Restrict only native MTP proposal readout to a calibration vocabulary")
    dispatch = parser.add_mutually_exclusive_group()
    dispatch.add_argument("--streamed-from", type=int, choices=range(2, 7),
                        help="Experiment: use existing 4-bit streamed affine kernels from this token count (stock: 6)")
    dispatch.add_argument("--tiled-from", type=int, choices=range(2, 6),
                          help="Experiment: use existing two-token affine tiles for short blocks from this token count")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/qwen3.5-9b-mlx-4bit")
    parser.add_argument("--draft-path", type=Path, default=ROOT / "models/qwen3.5-9b-mtp-4bit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.compare_stock_draft and args.draft_vocab is None:
        parser.error("--compare-stock-draft requires --draft-vocab")
    extra = {"compare_stock_draft": True} if args.compare_stock_draft else {}
    try:
        config = SweepConfig(args.prompt_count, args.tokens, args.repeats, tuple(args.block_sizes), args.trace)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output or ROOT / "artifacts/optimizations" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-mtp-sweep")
    if args.tiled_from is not None:
        from .tiled_affine import ScopedTiledAffine
        with ScopedTiledAffine(args.tiled_from) as optimization:
            return MTPSweepRunner(config, args.prompts, output, args.model_path, args.draft_path,
                                  optimization_metadata=optimization.metadata(), draft_vocab_path=args.draft_vocab, **extra).run()
    if args.streamed_from is not None:
        from .streamed_affine import ScopedStreamedAffine
        with ScopedStreamedAffine(args.streamed_from) as optimization:
            return MTPSweepRunner(config, args.prompts, output, args.model_path, args.draft_path,
                                  optimization_metadata=optimization.metadata(), draft_vocab_path=args.draft_vocab, **extra).run()
    return MTPSweepRunner(config, args.prompts, output, args.model_path, args.draft_path,
                          draft_vocab_path=args.draft_vocab, **extra).run()
