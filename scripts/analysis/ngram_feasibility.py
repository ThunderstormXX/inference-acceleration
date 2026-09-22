"""Replay n-gram drafts on saved trajectories without importing MLX or Torch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != DIRECTORY]
sys.path.insert(0, str(ROOT / "src"))

from inference_lab.optimizations.ngram import NgramFeasibilityAnalyzer


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_saved_traces(paths: list[Path]) -> dict:
    chains, provenance, seen = [], [], set()
    for path in paths:
        payload = path.read_bytes()
        summary_path = path.with_name("summary.json")
        summary_payload = summary_path.read_bytes()
        summary = json.loads(summary_payload)
        if summary.get("status") != "completed" or summary.get("samples_sha256") != digest(payload):
            raise ValueError(f"incomplete or modified source: {path}")
        provenance.append({"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                           "samples_sha256": digest(payload), "summary_sha256": digest(summary_payload),
                           "target_manifest_sha256": summary["sources"]["model_manifest_sha256"],
                           "data_manifest_sha256": summary["sources"]["data_manifest_sha256"]})
        for line in payload.splitlines():
            row = json.loads(line)
            if row.get("method") not in {"confidence_collect", "ar_before"}:
                continue
            # The heldout file also has a confidence run; prefer its actual AR.
            if "ar_before" in summary.get("methods", []) and row["method"] != "ar_before":
                continue
            index = row["source_row_index"]
            if type(index) is not int or index in seen:
                raise ValueError("duplicate or invalid source row")
            seen.add(index)
            if not row.get("host_observation", {}).get("valid"):
                raise ValueError("source host observation invalid")
            ids = row["generated_token_ids"]
            if ids != row["generation_trace"]["token_ids"] or len(ids) != row["generated_tokens"]:
                raise ValueError("source generation IDs inconsistent")
            chains.append({"source_row_index": index, "source_method": row["method"],
                           "prompt_token_ids": row["prompt_token_ids"], "generated_token_ids": ids})
    if not chains or len({item["target_manifest_sha256"] for item in provenance}) != 1:
        raise ValueError("need nonempty traces from one target")
    return {"schema_version": 1, "provenance": provenance, "chains": sorted(chains, key=lambda row: row["source_row_index"])}


def markdown(report: dict) -> str:
    lines = ["# Causal n-gram feasibility on saved Qwen trajectories", "",
             "CPU-only replay; **no live verifier and no measured inference speedup**.", "",
             f"{report['chains']} unique trajectories, {report['total_output_tokens']} output tokens. "
             "Each prompt is available at the start; only already committed output is added. "
             "The first generated token is treated as the prefill seed and excluded from decode positions.", "",
             "Rows 100–109 come from the earlier confidence collection; row 110 uses its AR-before output. "
             "All were previously inspected, so this is development material, not an unseen test set.", "",
             "The proposer chooses the longest suffix (up to 16 tokens), then the latest earlier occurrence "
             "with a known continuation. It never reads another chain or extrapolates beyond its committed history. "
             "Later true tokens are used only to label the already-issued proposal.", "",
             "## Every-position diagnostic", "",
             "Overlapping positions are dependent; the same tokens can contribute to several candidate windows. "
             "Coverage is the fraction of decode positions with a proposal. Accuracy is the first-token "
             "accuracy conditional on a proposal; accepted length counts only the uninterrupted correct prefix.", "",
             "| Minimum suffix | Max draft | Coverage | First correct | Mean accepted prefix | All offered accepted |",
             "|---:|---:|---:|---:|---:|---:|"]
    for config in report["configurations"]:
        row = config["aggregate"]["all_positions"]
        lines.append(f"| {config['min_match']} | {config['width']} | {row['opportunity_fraction']:.1%} | "
                     f"{row['first_token_accuracy_given_opportunity']:.1%} | {row['mean_accepted_prefix_per_opportunity']:.3f} | "
                     f"{row['fully_accepted_proposal_fraction']:.1%} |")
    lines += ["", "## Simulated causal rounds", "",
              "After each lookup the replay commits the matching prefix plus one correction/bonus token, capped "
              "at the budget. No match commits one AR token. These counts describe a hypothetical exact verifier; "
              "its batched numerical behavior, GDN rollback, GPU cost and Python/Metal synchronization are unmeasured. "
              "**Tokens per simulated round is not a speedup**, since wider rounds cost more.", "",
              "| Minimum suffix | Max draft | Rounds | Draft rounds | Tokens / round | First correct | CPU lookup µs/query |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for config in report["configurations"]:
        row = config["aggregate"]["simulated_rounds"]
        lines.append(f"| {config['min_match']} | {config['width']} | {row['visited_positions']} | {row['draft_opportunities']} | "
                     f"{row['tokens_per_simulated_round']:.3f} | {row['first_token_accuracy_given_opportunity']:.1%} | "
                     f"{row['cpu_propose_microseconds_per_query']:.2f} |")
    lines += ["", "CPU figures time `propose` only in one replay pass, excluding indexing updates, token transport "
              "and all GPU work. They are descriptive and are not subtracted from or converted into GPU savings.", "",
              "The JSON includes per-chain metrics, accepted-prefix histograms and bounded examples of full/partial/zero "
              "acceptance. No best hyperparameter is selected from these labels.", "",
              "## Reproduction", "", "```bash",
              "bash scripts/analysis/ngram_feasibility.sh --bundle docs/results/ngram-feasibility-inputs-2026-09-22.json.gz",
              "```", "", "The bundle contains only prompt/output token IDs and original source hashes. "
              "Run without `--bundle` to read the original local experiment artifacts. "
              "`--widths 1 2 4 --min-matches 2 4 8 --max-match 16` are the defaults.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--samples", nargs="+", type=Path, default=[
        ROOT / "artifacts/experiments/mtp-confidence/calibration2048/samples.jsonl",
        ROOT / "artifacts/experiments/mtp-confidence/heldout2048/samples.jsonl"])
    parser.add_argument("--widths", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--min-matches", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--max-match", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/ngram-feasibility-2026-09-22.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/ngram-feasibility-2026-09-22.md")
    parser.add_argument("--export-bundle", type=Path, default=ROOT / "docs/results/ngram-feasibility-inputs-2026-09-22.json.gz")
    args = parser.parse_args(argv)
    if args.bundle:
        payload = args.bundle.read_bytes()
        bundle = json.loads(gzip.decompress(payload) if args.bundle.suffix == ".gz" else payload)
    else:
        bundle = load_saved_traces([path.resolve() for path in args.samples])
    chains = bundle["chains"]
    if not chains or len({row["source_row_index"] for row in chains}) != len(chains):
        raise ValueError("bundle must contain unique nonempty chains")
    compact = json.dumps(bundle, separators=(",", ":"), sort_keys=True).encode()
    configurations = []
    for minimum in args.min_matches:
        for width in args.widths:
            analyzer = NgramFeasibilityAnalyzer(width=width, min_match=minimum, max_match=args.max_match)
            per_chain = []
            for row in chains:
                result = {"source_row_index": row["source_row_index"], "source_method": row["source_method"]}
                for mode in ("all_positions", "simulated_rounds"):
                    result[mode] = analyzer.analyze(row["prompt_token_ids"], row["generated_token_ids"], mode=mode)
                per_chain.append(result)
            configurations.append({"width": width, "min_match": minimum, "max_match": args.max_match,
                                   "per_chain": per_chain, "aggregate": {
                                       mode: analyzer.aggregate([row[mode] for row in per_chain])
                                       for mode in ("all_positions", "simulated_rounds")}})
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "kind": "cpu_causal_replay_not_live_inference", "chains": len(chains),
              "total_output_tokens": sum(len(row["generated_token_ids"]) for row in chains),
              "input_sha256": digest(compact), "provenance": bundle["provenance"],
              "proposer_code_sha256": digest((ROOT / "src/inference_lab/optimizations/ngram.py").read_bytes()),
              "protocol": {"history": "this prompt plus committed tokens of this chain only", "seed_output_tokens": 1,
                           "static_corpus": None, "hyperparameter_selection": None,
                           "labels": "future output used only after issuing each proposal",
                           "limitation": "No inference speedup or live numerical parity measured; all chains are development material."},
              "configurations": configurations}
    for path in (args.output, args.report, args.export_bundle):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.export_bundle.write_bytes(gzip.compress(compact, mtime=0))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.report.write_text(markdown(report))
    print(f"Replayed {len(chains)} chains / {report['total_output_tokens']} tokens; no GPU imports or inference.")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
