"""Combine bounded speed experiments without importing inference runtimes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def _stats(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)} if values else None


def _all(values):
    values = list(values)
    return False if False in values else True if values and all(v is True for v in values) else None


def _read(path):
    payload = path.read_bytes()
    decoded = gzip.decompress(payload) if path.suffix == ".gz" else payload
    result = json.loads(decoded, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON number {value}")))
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result, {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}


def _speed(row):
    measure = row.get("measurement", {})
    tokens, seconds = measure.get("decode_tokens"), measure.get("decode_seconds")
    if (type(tokens) is not int or tokens <= 0 or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("invalid decode token count or duration")
    return tokens / seconds


def _key(row, family):
    return row.get("variant") if family in {"kernel_sweep", "ngram_sweep"} else f"{row.get('mode')}_k{row.get('block_size')}"


def _raw_drift_bracket(runs, family):
    if family == "kernel_sweep":
        before = [r for r in runs if r.get("variant") == "ar" and r.get("prompt_index") == 0 and r.get("phase") == "reference"]
        after = [r for r in runs if r.get("variant") == "ar" and r.get("prompt_index") == 0 and r.get("phase") == "bracket"]
        pair = [before[0], after[-1]] if before and after else []
    else:
        baseline_key = "ar_k1" if family == "mtp_sweep" else "ar"
        brackets = [r for r in runs if _key(r, family) == baseline_key and r.get("prompt_index") == 0 and r.get("role") == "bracket"]
        pair = [brackets[0], brackets[-1]] if len(brackets) >= 2 else []
    if not pair:
        return None
    try:
        before_rate, after_rate = [_speed(row) for row in pair]
    except ValueError:
        return None
    return {"prompt_index": 0, "before_tok_s": before_rate, "after_tok_s": after_rate,
            "after_over_before": after_rate / before_rate, "source": "raw AR bracket measurements"}


def _direct_draft_comparisons(measured):
    comparisons, warnings = [], []
    for stock in (row for row in measured if row.get("mode") == "mtp-stock"):
        treatments = [row for row in measured if row.get("mode") == "mtp" and all(
            row.get(key) == stock.get(key) for key in ("prompt_index", "repeat", "block_size"))]
        if len(treatments) != 1:
            warnings.append("missing or ambiguous shortlist/stock MTP match")
            continue
        treatment = treatments[0]
        try:
            ratio = _speed(treatment) / _speed(stock)
        except ValueError:
            warnings.append("invalid shortlist/stock MTP duration")
            continue
        if treatment["measurement"]["decode_tokens"] != stock["measurement"]["decode_tokens"]:
            warnings.append("shortlist/stock MTP token budgets differ")
            continue
        a, b = treatment["measurement"].get("generated_token_ids"), stock["measurement"].get("generated_token_ids")
        comparisons.append({"prompt_index": stock.get("prompt_index"), "repeat": stock.get("repeat"),
                            "block_size": stock.get("block_size"), "shortlist_over_stock": ratio,
                            "token_ids_match": a == b if isinstance(a, list) and isinstance(b, list) else None})
    return comparisons, warnings


def read_experiment(directory: Path) -> dict:
    """Keep failures as explicit records; incomplete sources never qualify."""
    result = {"name": directory.name, "source_directory": str(directory.resolve()),
              "status": "unavailable", "family": None, "protocol": {}, "provenance": [],
              "warnings": [], "errors": [], "rows": [], "ranking": None, "diagnostic_flags": []}
    try:
        summary, stamp = _read(directory / "summary.json")
        result["provenance"].append(stamp)
    except (OSError, ValueError) as exc:
        summary = {}
        result["warnings"].append(f"summary unavailable: {exc}")
    raw_path = next((directory / name for name in ("raw.json.gz", "raw.json") if (directory / name).exists()), None)
    try:
        raw, stamp = _read(raw_path) if raw_path else ({}, None)
        if stamp:
            result["provenance"].append(stamp)
    except (OSError, ValueError, EOFError) as exc:
        raw = {}
        result["warnings"].append(f"raw data unavailable: {exc}")
    protocol = raw.get("protocol", {})
    if isinstance(protocol.get("widths"), list) and isinstance(protocol.get("min_matches"), list):
        family = "ngram_sweep"
    elif "by_variant" in summary:
        family = "ngram_sweep" if any(isinstance(row, dict) and row.get("role") == "paired" and "variant" in row for row in raw.get("runs", [])) else "kernel_sweep"
    else:
        family = "mtp_sweep" if "by_mode" in summary else None
    if family is None and raw.get("runs"):
        family = "kernel_sweep" if "variant" in raw["runs"][0] else "mtp_sweep"
    result.update(family=family, status=raw.get("status", summary.get("status", "unavailable")),
                  protocol=raw.get("protocol", {}),
                  backend=raw.get("backend"), baseline=raw.get("baseline"),
                  drift_bracket=summary.get("drift_bracket"))
    if summary.get("status") and raw.get("status") and summary["status"] != raw["status"]:
        result["warnings"].append("summary/raw status differ; snapshot may be from an active run")
    errors = raw.get("errors", [])
    result["errors"] = errors if isinstance(errors, list) else [errors]
    if raw.get("error"):
        result["errors"].append({"error": raw["error"], "error_type": raw.get("error_type")})
    if summary.get("errors") and not result["errors"]:
        result["errors"].append({"summary_error_count": summary["errors"]})
    if family is None:
        result["warnings"].append("unknown or empty experiment format")
        return result
    runs = raw.get("runs", [])
    if not isinstance(runs, list) or any(not isinstance(row, dict) for row in runs):
        result["warnings"].append("invalid raw runs")
        runs = []
    raw_drift = _raw_drift_bracket(runs, family)
    if raw_drift is not None:
        result["drift_bracket"] = raw_drift
    drift = result.get("drift_bracket") or {}
    ratio = drift.get("after_over_before")
    result["drift_diagnostic_threshold_fraction"] = 0.05
    if isinstance(ratio, (int, float)) and math.isfinite(ratio) and abs(ratio - 1) > 0.05:
        message = f"AR bracket drift {100 * (ratio - 1):+.1f}% exceeds 5%; diagnostic only"
        result["warnings"].append(message)
        result["diagnostic_flags"].append(message)
    measured = [row for row in runs if row.get("phase") == "measurement"] if family == "kernel_sweep" else [row for row in runs if row.get("role") == "paired"]
    summary_rows = summary.get("by_mode" if family == "mtp_sweep" else "by_variant", {})
    names = set(summary_rows) | {_key(row, family) for row in measured}
    if family == "kernel_sweep":
        names.update(raw.get("protocol", {}).get("variants", []))
    elif family == "mtp_sweep":
        names.update(f"mtp_k{k}" for k in raw.get("protocol", {}).get("block_sizes", []))
        if protocol.get("compare_stock_draft") is True:
            names.update(f"mtp-stock_k{k}" for k in protocol.get("block_sizes", []))
    else:
        names.update(f"ngram-w{width}-m{minimum}" for minimum in protocol.get("min_matches", [])
                     for width in protocol.get("widths", []))
    result["all_run_token_parity"] = _all(row.get("matches_baseline") for row in runs)
    result["all_runs_awake"] = _all(not row["sleep"]["sleep_detected"] if isinstance(row.get("sleep", {}).get("sleep_detected"), bool) else None for row in runs)
    result["completed_raw_runs"] = len(runs)
    result["expected_raw_runs"] = raw.get("protocol", {}).get("expected_runs")
    protocol = raw.get("protocol", {})
    prompt_count = len(protocol["prompts"]) if isinstance(protocol.get("prompts"), list) else None
    repeats = protocol.get("repeats")
    expected_per_variant = prompt_count * repeats if prompt_count and type(repeats) is int and repeats > 0 else None
    if result["expected_raw_runs"] is None and family == "kernel_sweep" and expected_per_variant is not None:
        result["expected_raw_runs"] = expected_per_variant * len(protocol.get("variants", [])) + prompt_count + 1
    if result["expected_raw_runs"] is None and family == "ngram_sweep" and expected_per_variant is not None:
        configurations = len(protocol.get("widths", [])) * len(protocol.get("min_matches", []))
        if configurations:
            # One adjacent AR per config, two p0 brackets, other prompt references.
            result["expected_raw_runs"] = 2 * expected_per_variant * configurations + prompt_count + 1
    if result["expected_raw_runs"] is not None and result["expected_raw_runs"] != len(runs):
        result["warnings"].append("completed raw run count does not match the requested protocol")
    if family == "mtp_sweep":
        comparisons, comparison_warnings = _direct_draft_comparisons(measured)
        result["warnings"].extend(comparison_warnings)
        result["draft_comparisons"] = comparisons
        result["shortlist_over_stock"] = _stats([row["shortlist_over_stock"] for row in comparisons])
        result["reported_draft_comparisons"] = summary.get("draft_comparisons")
        result["reported_shortlist_over_stock"] = summary.get("shortlist_over_stock")
        reported = result["reported_shortlist_over_stock"]
        computed = result["shortlist_over_stock"]
        if reported and computed and (reported.get("n") != computed["n"] or not math.isclose(reported.get("mean", float("nan")), computed["mean"], rel_tol=1e-8)):
            result["warnings"].append("summary/raw shortlist comparison differs; raw measurements used")
        result["draft_comparison_by_block_size"] = {}
        for k in sorted({row["block_size"] for row in comparisons}):
            rows = [row for row in comparisons if row["block_size"] == k]
            result["draft_comparison_by_block_size"][str(k)] = {
                "shortlist_over_stock": _stats([row["shortlist_over_stock"] for row in rows]),
                "paired_measurements": len(rows),
                "unique_prompt_count": len({row["prompt_index"] for row in rows}),
                "all_token_ids_match": _all(row["token_ids_match"] for row in rows)}
    for name in sorted(name for name in names if isinstance(name, str)):
        selected = [row for row in measured if _key(row, family) == name]
        warnings, valid = [], []
        for row in selected:
            try:
                _speed(row)
                valid.append(row)
            except ValueError as exc:
                warnings.append(str(exc))
        rates = [_speed(row) for row in valid]
        baseline_name = "ar_k1" if family == "mtp_sweep" else "ar"
        ratios, pair_parity, matched = [], [], 0
        if name != baseline_name:
            for row in valid:
                candidates = [base for base in measured if _key(base, family) == baseline_name and (
                    (base.get("prompt_index"), base.get("repeat")) == (row.get("prompt_index"), row.get("repeat"))
                    if family == "kernel_sweep" else base.get("pair_id") is not None and base.get("pair_id") == row.get("pair_id"))]
                if len(candidates) != 1:
                    warnings.append("missing or ambiguous matched AR")
                    continue
                base = candidates[0]
                try:
                    baseline_rate = _speed(base)
                except ValueError:
                    warnings.append("invalid matched AR duration")
                    continue
                if base["measurement"]["decode_tokens"] != row["measurement"]["decode_tokens"]:
                    warnings.append("matched AR has a different token budget")
                    continue
                ratios.append(_speed(row) / baseline_rate)
                a, b = row["measurement"].get("generated_token_ids"), base["measurement"].get("generated_token_ids")
                pair_parity.append(a == b if isinstance(a, list) and isinstance(b, list) else None)
                matched += 1
        expected = expected_per_variant
        if family == "mtp_sweep" and name == "ar_k1" and expected is not None:
            expected *= len(protocol.get("block_sizes", []))
            if protocol.get("compare_stock_draft") is True:
                expected *= 2
        if family == "ngram_sweep" and name == "ar" and expected is not None:
            configurations = len(protocol.get("widths", [])) * len(protocol.get("min_matches", []))
            expected = expected * configurations if configurations else None
        if expected is not None and len(selected) != expected:
            warnings.append(f"measured {len(selected)} of {expected} requested runs")
        stats = _stats(rates)
        source_stats = summary_rows.get(name, {}).get("tok_s")
        if source_stats and stats and (source_stats.get("n") != stats["n"] or not math.isclose(source_stats.get("mean", float("nan")), stats["mean"], rel_tol=1e-8)):
            warnings.append("summary/raw statistics differ; raw measurements used")
        parity = _all(row.get("matches_baseline") for row in selected)
        awake = _all(not row["sleep"]["sleep_detected"] if isinstance(row.get("sleep", {}).get("sleep_detected"), bool) else None for row in selected)
        no_claims = list(result["diagnostic_flags"])
        if result["status"] != "completed": no_claims.append(f"experiment status {result['status']}")
        if result["errors"]: no_claims.append("experiment contains errors")
        if result["warnings"] or warnings: no_claims.append("source validation warnings")
        if not selected: no_claims.append("no raw measurements")
        if parity is not True: no_claims.append("token parity failed" if parity is False else "token parity unknown")
        if awake is not True: no_claims.append("sleep detected" if awake is False else "sleep status unknown")
        if result["all_run_token_parity"] is not True: no_claims.append("run-wide parity not confirmed")
        if result["all_runs_awake"] is not True: no_claims.append("run-wide awake status not confirmed")
        if name != baseline_name and (matched != len(selected) or _all(pair_parity) is not True):
            no_claims.append("not all measurements have an exact matched AR pair")
        result["rows"].append({"name": name, "role": "control" if name == baseline_name else "experimental",
            "tok_s": stats or source_stats, "statistics_source": "raw" if stats else "summary_only",
            "paired_speed_ratio": _stats(ratios), "paired_measurements": matched,
            "measurement_count": len(selected) if raw else None,
            "unique_prompt_count": len({r.get("prompt_index") for r in selected}) if selected else None,
            "repetition_ids": sorted({r["repeat"] for r in selected if isinstance(r.get("repeat"), int)}),
            "all_token_ids_match": parity, "all_runs_awake": awake,
            "eligible_for_same_protocol_comparison": not no_claims,
            "no_claim_reasons": no_claims, "warnings": warnings,
            "reported_summary": summary_rows.get(name)})
    for k, comparison in result.get("draft_comparison_by_block_size", {}).items():
        methods = [row for row in result["rows"] if row["name"] in {f"mtp_k{k}", f"mtp-stock_k{k}"}]
        comparison["eligible_for_same_protocol_comparison"] = (
            len(methods) == 2 and all(row["eligible_for_same_protocol_comparison"] for row in methods)
            and comparison["all_token_ids_match"] is True
            and all(row["measurement_count"] == comparison["paired_measurements"] for row in methods))
    return result


def render_markdown(report: dict) -> str:
    def shown(value):
        return "PASS" if value is True else "FAIL" if value is False else "unknown"
    def number(stats):
        if not stats or not isinstance(stats.get("mean"), (int, float)):
            return "—"
        return f"{stats['mean']:.3f} ± {stats['stdev']:.3f}" if stats.get("stdev") is not None else f"{stats['mean']:.3f} (SD n/a)"
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# Local speed experiments", "", "Experimental measurements, separated by protocol; no cross-experiment ranking.", "",
             "Numbers are mean ± sample SD across measured requests. Repeated prompts are dependent: request count "
             "is **not** a count of independent tasks. SD describes these runs, not uncertainty over a task population. "
             "Missing SD for a single observation is not zero variance. References/brackets are excluded from means.", "",
             "A failed/unknown parity, sleep event, incomplete run or unresolved source warning prevents a speedup claim. "
             "Reported rates remain visible for diagnosis. An absolute AR start/end bracket drift above 5% also makes "
             "the experiment diagnostic, without invalidating its raw measurements. Even a valid comparison is empirical for these prompts.", ""]
    for experiment in report["experiments"]:
        protocol = experiment["protocol"]
        lines += [f"## {escape(experiment['name'])}", "",
                  f"Family: `{experiment['family']}`; status: **{escape(experiment['status'])}**. "
                  f"Output budget: {protocol.get('tokens', 'unknown')}; requested repeats: {protocol.get('repeats', 'unknown')}. "
                  f"Trace enabled: {protocol.get('trace_generation', protocol.get('trace', 'unknown'))}.", ""]
        if protocol.get("block_size") is not None:
            lines.append(f"DFlash block size: {protocol['block_size']}.\n")
        if experiment["family"] == "ngram_sweep":
            lines.append(f"N-gram proposal widths: {protocol.get('widths', 'unknown')}; minimum suffixes: {protocol.get('min_matches', 'unknown')}. Each variant is paired with its neighboring AR.\n")
        if protocol.get("notes"):
            lines.extend(f"- {escape(note)}" for note in protocol["notes"])
            lines.append("")
        lines += ["| Variant | Decode tok/s | Paired ratio vs AR | Runs | Unique prompts | Observed repeats | Parity | Awake | Status |",
                  "|---|---:|---:|---:|---:|---|---|---|---|"]
        for row in experiment["rows"]:
            state = "diagnostic only: " + "; ".join(row["no_claim_reasons"]) if row["no_claim_reasons"] else "control" if row["role"] == "control" else "validated experimental comparison"
            lines.append(f"| {escape(row['name'])} | {number(row['tok_s'])} | {number(row['paired_speed_ratio'])} | "
                         f"{row['measurement_count'] if row['measurement_count'] is not None else 'unknown'} | "
                         f"{row['unique_prompt_count'] if row['unique_prompt_count'] is not None else 'unknown'} | "
                         f"{escape(row['repetition_ids'])} | {shown(row['all_token_ids_match'])} | {shown(row['all_runs_awake'])} | {escape(state)} |")
        if not experiment["rows"]:
            lines.append("| No measurements | — | — | — | — | — | unknown | unknown | diagnostic only |")
        lines.append("")
        if protocol.get("compare_stock_draft") is True or experiment.get("draft_comparisons"):
            lines += ["Direct draft comparison: shortlist MTP versus stock full-vocabulary draft, matched by "
                      "prompt, repeat and block size. This ratio is separate from each method's neighboring AR control.", "",
                      "| Block size | Shortlist / stock ratio | Pairs | Unique prompts | Parity | Status |",
                      "|---:|---:|---:|---:|---|---|"]
            direct = experiment.get("draft_comparison_by_block_size", {})
            for k, comparison in direct.items():
                state = "validated experimental comparison" if comparison["eligible_for_same_protocol_comparison"] else "diagnostic only"
                lines.append(f"| {k} | {number(comparison['shortlist_over_stock'])} | {comparison['paired_measurements']} | "
                             f"{comparison['unique_prompt_count']} | {shown(comparison['all_token_ids_match'])} | {state} |")
            if not direct:
                lines.append("| — | — | 0 | — | unknown | diagnostic only: no complete matches |")
            lines.append("")
        for warning in experiment["warnings"]:
            lines.append(f"- Source warning: {escape(warning)}")
        if experiment["errors"]:
            lines.append(f"- Recorded errors: {len(experiment['errors'])}; details preserved in JSON.")
        if experiment.get("drift_bracket"):
            lines.append(f"- AR drift bracket: `{json.dumps(experiment['drift_bracket'], sort_keys=True)}`.")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output stem; .json and .md are appended")
    args = parser.parse_args(argv)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "interpretation": "Experimental per-protocol comparisons, no pooled cross-protocol ranking or population uncertainty.",
              "experiments": [read_experiment(path) for path in args.inputs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.output) + ".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    Path(str(args.output) + ".md").write_text(render_markdown(report))
    print(f"Saved {len(report['experiments'])} protocol-separated experiments to {args.output}.json/.md")
    return 0
