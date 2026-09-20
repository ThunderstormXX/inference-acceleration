"""Deterministic, heldout-free calibration of a native-MTP confidence gate."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile

from inference_lab.visualization.recording import validate_generation_trace


THRESHOLDS = tuple(index / 20 for index in range(21))
EXPECTED_INDICES = tuple(range(100, 110))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _dispersion(values):
    values = list(values)
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "sample_variance": statistics.variance(values) if len(values) > 1 else None,
            "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def _classification(rounds, threshold):
    tp = fp = tn = fn = rejected_first = rejected_second = 0
    for record in rounds:
        selected = record["score"] >= threshold
        positive = record["accepted_count"] == 2
        tp += selected and positive
        fp += selected and not positive
        tn += not selected and not positive
        fn += not selected and positive
        rejected_first += selected and record["accepted_count"] == 0
        rejected_second += selected and record["accepted_count"] == 1
    recall, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    class_recalls = [value for value in (recall, specificity) if value is not None]
    total, selected = tp + fp + tn + fn, tp + fp
    return {
        "rounds": total, "selected_rounds": selected,
        "confusion": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "positive_rounds": tp + fn, "negative_rounds": tn + fp,
        "classes_present": len(class_recalls),
        "coverage": _ratio(selected, total), "precision": _ratio(tp, selected),
        "recall": recall, "specificity": specificity, "accuracy": _ratio(tp + tn, total),
        "balanced_accuracy": statistics.mean(class_recalls) if class_recalls else None,
        "selected_rejected_first_rounds": rejected_first,
        "selected_rejected_second_rounds": rejected_second,
        "selected_rejected_first_fraction": _ratio(rejected_first, selected),
        "selected_rejected_second_fraction": _ratio(rejected_second, selected),
    }


def _acceptance(rounds):
    first = sum(record["accepted_count"] >= 1 for record in rounds)
    both = sum(record["accepted_count"] == 2 for record in rounds)
    return {"rounds": len(rounds), "first_accepted_rounds": first, "both_accepted_rounds": both,
            "first_acceptance_rate": _ratio(first, len(rounds)),
            "second_acceptance_given_first_rate": _ratio(both, first),
            "both_acceptance_rate": _ratio(both, len(rounds)),
            "rejected_first_rounds": len(rounds) - first, "rejected_second_rounds": first - both}


def _reliability(pairs):
    """Descriptive score reliability; these scores are not accepted as calibrated probabilities."""
    bins = []
    weighted_gap = 0.0
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        values = [(score, outcome) for score, outcome in pairs
                  if lower <= score and (score < upper or (index == 9 and score <= upper))]
        mean_score = statistics.mean(score for score, _ in values) if values else None
        frequency = statistics.mean(outcome for _, outcome in values) if values else None
        gap = abs(mean_score - frequency) if values else None
        weighted_gap += len(values) * gap if gap is not None else 0.0
        bins.append({"lower": lower, "upper": upper, "upper_inclusive": index == 9,
                     "count": len(values), "mean_score": mean_score,
                     "observed_acceptance_rate": frequency, "absolute_gap": gap})
    return {"count": len(pairs),
            "brier_score": statistics.mean((score - outcome) ** 2 for score, outcome in pairs) if pairs else None,
            "expected_calibration_error_10_bins": _ratio(weighted_gap, len(pairs)), "bins": bins}


class ConfidenceCalibrator:
    """Choose a fixed threshold on complete calibration chains only, without GPU imports."""

    def __init__(self, samples_path, expected_indices=EXPECTED_INDICES, *, expected_max_new_tokens=2048):
        self.samples_path = Path(samples_path).resolve()
        self.expected_indices = tuple(expected_indices)
        _require(bool(self.expected_indices) and all(type(index) is int and index >= 0 for index in self.expected_indices),
                 "Expected calibration indices must be nonnegative integers")
        _require(len(set(self.expected_indices)) == len(self.expected_indices), "Duplicate expected calibration indices")
        _require(110 not in self.expected_indices, "Heldout source index 110 must never enter calibration")
        self.expected_indices = tuple(sorted(self.expected_indices))
        _require(type(expected_max_new_tokens) is int and expected_max_new_tokens >= 3,
                 "expected_max_new_tokens must be an integer >= 3")
        self.expected_max_new_tokens = expected_max_new_tokens

    @staticmethod
    def _probability(value, name):
        _require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                 f"Invalid probability: {name}")
        return float(value)

    def _load(self):
        payload = self.samples_path.read_bytes()
        lines = payload.splitlines()
        _require(len(lines) == len(self.expected_indices) and all(line.strip() for line in lines),
                 "Calibration requires exactly one complete row per expected source index")
        rows = [json.loads(line) for line in lines]
        _require(all(isinstance(row, dict) for row in rows), "Calibration samples must be objects")
        indices = [row.get("source_row_index") for row in rows]
        _require(all(type(index) is int for index in indices) and sorted(indices) == list(self.expected_indices),
                 "Calibration source indices differ, are duplicated, or include heldout/leaked rows")
        summary_path = self.samples_path.parent / "summary.json"
        summary_payload = summary_path.read_bytes()
        summary = json.loads(summary_payload)
        _require(isinstance(summary, dict), "Calibration summary must be an object")
        config = summary.get("config", {})
        _require(summary.get("status") == "completed" and config.get("stage") == "collect"
                 and summary.get("methods") == ["confidence_collect"],
                 "Calibration requires a completed collect-only summary")
        _require(type(config.get("max_new_tokens")) is int and config["max_new_tokens"] == self.expected_max_new_tokens,
                 "Calibration output budget differs from expected_max_new_tokens")
        _require(summary.get("samples_sha256") == _sha(payload), "Calibration samples hash differs from completed summary")
        _require(type(summary.get("completed_measurements")) is int
                 and summary["completed_measurements"] == len(rows)
                 and summary.get("source_row_indices") == list(self.expected_indices),
                 "Calibration summary coverage differs from expected source indices")
        sources = summary.get("sources")
        _require(isinstance(sources, dict), "Calibration summary requires source provenance")
        for name in ("data_manifest_sha256", "model_manifest_sha256", "draft_manifest_sha256"):
            _require(isinstance(sources.get(name), str) and re.fullmatch(r"[0-9a-f]{64}", sources[name]),
                     f"Missing or invalid calibration provenance: {name}")
        _require(sources.get("policy_sha256") is None, "Collect data must precede any selected calibration policy")
        chains, excluded = {}, {}
        for row in sorted(rows, key=lambda item: item["source_row_index"]):
            index = row["source_row_index"]
            _require(row.get("method") == "confidence_collect", f"Row {index} is not a collect measurement")
            _require(len(row.get("generated_token_ids", [])) == self.expected_max_new_tokens,
                     f"Row {index} is partial: expected {self.expected_max_new_tokens} output IDs")
            validate_generation_trace(row)
            _require(isinstance(row.get("host_observation"), dict) and row["host_observation"].get("valid") is True,
                     f"Row {index} lacks valid host timing/power evidence")
            records = row.get("confidence_rounds")
            _require(isinstance(records, list) and bool(records), f"Row {index} has no confidence rounds")
            events = row["generation_trace"]["events"]
            drafts = [event for event in events if event["type"] == "draft"]
            commits = [event for event in events if event["type"] == "commit"][1:]
            _require(len(records) == len(drafts) == len(commits),
                     f"Row {index}: confidence rounds do not cover every traced decode round")
            eligible, tails = [], 0
            position = 1
            for number, (record, draft, commit) in enumerate(zip(records, drafts, commits), 1):
                _require(isinstance(record, dict) and record.get("decision") == "verify",
                         f"Row {index}: calibration may contain only fully verified rounds")
                probabilities = record.get("proposal_probabilities")
                _require(isinstance(probabilities, list) and 1 <= len(probabilities) <= 2,
                         f"Row {index}: expected one or two proposal probabilities")
                probs = [self._probability(value, "proposal") for value in probabilities]
                accepted = record.get("accepted_count")
                _require(type(accepted) is int and 0 <= accepted <= len(probs),
                         f"Row {index}: invalid accepted_count")
                _require(type(record.get("output_count_before")) is int and record["output_count_before"] == position
                         and draft["output_count"] == position and len(draft["token_ids"]) == len(probs)
                         and commit["accepted_count"] == accepted,
                         f"Row {index}: confidence counters differ from raw trace")
                _require(len(probs) == min(2, self.expected_max_new_tokens - position),
                         f"Row {index}: shortened proposal round before output-budget tail")
                _require(len(commit["token_ids"]) == min(accepted + 1, self.expected_max_new_tokens - position),
                         f"Row {index}: committed output count differs from greedy acceptance")
                for name, expected in (("proposal_token_ids", draft["token_ids"]),
                                       ("emitted_token_ids", commit["token_ids"]),
                                       ("verified_draft_count", len(probs)), ("round", draft.get("round"))):
                    if name in record:
                        _require(record[name] == expected, f"Row {index}: {name} differs from raw trace")
                score = self._probability(record.get("confidence_product"), "confidence_product")
                _require(math.isclose(score, math.prod(probs), rel_tol=1e-10, abs_tol=1e-15),
                         f"Row {index}: confidence_product differs from proposal probability product")
                if len(probs) == 2:
                    target_ids = record.get("target_token_ids", commit.get("target_token_ids"))
                    if target_ids is not None:
                        _require(isinstance(target_ids, list) and bool(target_ids)
                                 and all(type(token) is int and token >= 0 for token in target_ids),
                                 f"Row {index}: invalid target_token_ids")
                    if "target_token_ids" in record and "target_token_ids" in commit:
                        _require(record["target_token_ids"] == commit["target_token_ids"],
                                 f"Row {index}: target_token_ids differs from raw trace")
                    exported = {
                        "source_row_index": index, "round": record.get("round", number),
                        "output_count_before": position,
                        "previous_generated_token_ids": row["generated_token_ids"][max(0, position - 12):position],
                        "proposal_token_ids": list(draft["token_ids"]),
                        "p1": probs[0], "p2": probs[1], "confidence_product": score,
                        "accepted_count": accepted, "both_accepted": accepted == 2,
                        "first_accepted": accepted >= 1,
                        "second_accepted_given_first": accepted == 2 if accepted >= 1 else None,
                        "target_token_ids": target_ids, "emitted_token_ids": list(commit["token_ids"]),
                    }
                    eligible.append({"round": number, "score": score, "p1": probs[0], "p2": probs[1],
                                     "accepted_count": accepted, "export": exported})
                else:
                    tails += 1
                position = commit["output_count"]
            _require(position == self.expected_max_new_tokens, f"Row {index}: partial confidence trace")
            _require(bool(eligible), f"Row {index}: no valid full two-proposal rounds")
            chains[index], excluded[index] = eligible, tails
        return chains, excluded, {"source_samples_path": str(self.samples_path), "source_samples_sha256": _sha(payload),
                                  "source_summary_path": str(summary_path), "source_summary_sha256": _sha(summary_payload),
                                  "source_provenance": sources}

    @staticmethod
    def _grid(chains):
        scores = []
        for threshold in THRESHOLDS:
            metrics = [_classification(records, threshold) for records in chains.values()]
            balanced = _dispersion(item["balanced_accuracy"] for item in metrics)
            scores.append({"threshold": threshold, "macro_balanced_accuracy": balanced["mean"],
                           "per_chain_balanced_accuracy": balanced,
                           "macro_coverage": statistics.mean(item["coverage"] for item in metrics),
                           "single_class_chains": sum(item["classes_present"] == 1 for item in metrics)})
        # Sorted grid plus strict improvement gives deterministic lower-threshold tie breaking.
        best = scores[0]
        for candidate in scores[1:]:
            if candidate["macro_balanced_accuracy"] > best["macro_balanced_accuracy"] + 1e-15:
                best = candidate
        return scores, best

    def analyze(self):
        report, _ = self._analyze()
        return report

    def _analyze(self):
        chains, excluded, evidence = self._load()
        round_rows = [record["export"] for records in chains.values() for record in records]
        rounds_payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                                   + "\n").encode() for row in round_rows)
        rounds_dataset = {"schema_version": 1, "filename": "rounds.jsonl", "sha256": _sha(rounds_payload),
                          "count": len(round_rows), "unit": "verified_full_two_proposal_round"}
        grid, selected = self._grid(chains)
        threshold = selected["threshold"]
        pooled = [record for records in chains.values() for record in records]
        per_chain = []
        for index, records in chains.items():
            per_chain.append({"source_row_index": index, "eligible_rounds": len(records),
                              "excluded_one_proposal_tail_rounds": excluded[index],
                              "selected_threshold_metrics": _classification(records, threshold),
                              "observed_acceptance": _acceptance(records),
                              "product_brier_score": _reliability([(r["score"], int(r["accepted_count"] == 2)) for r in records])["brier_score"]})
        folds = []
        if len(chains) > 1:
            for index in chains:
                training = {key: records for key, records in chains.items() if key != index}
                _, best = self._grid(training)
                folds.append({"left_out_source_row_index": index, "training_source_row_indices": list(training),
                              "selected_threshold": best["threshold"], "training_macro_balanced_accuracy": best["macro_balanced_accuracy"],
                              "left_out_metrics": _classification(chains[index], best["threshold"])})
        counts = Counter(fold["selected_threshold"] for fold in folds)
        reliability = {
            "product_vs_both_accepted": _reliability([(r["score"], int(r["accepted_count"] == 2)) for r in pooled]),
            "p1_vs_first_accepted": _reliability([(r["p1"], int(r["accepted_count"] >= 1)) for r in pooled]),
            "p2_vs_second_accepted_given_first": _reliability([(r["p2"], int(r["accepted_count"] == 2))
                                                               for r in pooled if r["accepted_count"] >= 1]),
        }
        report = {
            "schema_version": 1, "status": "completed", **evidence,
            "rounds_dataset": rounds_dataset,
            "calibration_source_row_indices": list(chains), "expected_max_new_tokens": self.expected_max_new_tokens,
            "heldout_source_row_indices_excluded": [110], "selected_threshold": threshold,
            "selection": {"objective": "macro per-chain balanced accuracy for both proposals accepted",
                          "positive_label": "accepted_count == 2", "prediction": "confidence_product >= threshold",
                          "class_handling": "mean recalls over classes present within each chain; single-class chains are retained",
                          "tie_break": "lowest threshold; numerical ties within 1e-15",
                          "threshold_grid": list(THRESHOLDS), "random_seed": None,
                          "macro_balanced_accuracy": selected["macro_balanced_accuracy"]},
            "eligible_rounds": len(pooled), "excluded_one_proposal_tail_rounds": sum(excluded.values()),
            "selected_threshold_pooled_metrics": _classification(pooled, threshold),
            "observed_acceptance": _acceptance(pooled), "threshold_grid_results": grid, "per_chain": per_chain,
            "per_chain_dispersion": {
                "balanced_accuracy_at_selected_threshold": _dispersion(item["selected_threshold_metrics"]["balanced_accuracy"] for item in per_chain),
                "coverage_at_selected_threshold": _dispersion(item["selected_threshold_metrics"]["coverage"] for item in per_chain),
                "both_acceptance_rate": _dispersion(item["observed_acceptance"]["both_acceptance_rate"] for item in per_chain),
                "product_brier_score": _dispersion(item["product_brier_score"] for item in per_chain)},
            "score_reliability": reliability,
            "leave_one_chain_out": {"folds": folds,
                "threshold_distribution": [{"threshold": key, "folds": counts[key]} for key in sorted(counts)],
                "selected_threshold_dispersion": _dispersion(fold["selected_threshold"] for fold in folds),
                "left_out_balanced_accuracy": _dispersion(fold["left_out_metrics"]["balanced_accuracy"] for fold in folds)},
            "interpretation": {
                "confidence": "Product of drafter chosen-token probabilities; not a calibrated probability of target acceptance.",
                "objective": "Acceptance classification on calibration chains; this is not a speed optimizer.",
                "timing": "Offline traces cannot identify counterfactual gate execution time or speedup.",
                "dependence": "Rounds within a chain are dependent; objective averages chains equally and dispersion is descriptive, not a confidence interval.",
                "policy_scope": "Only full two-proposal rounds enter calibration; one-proposal budget tails are excluded. Frozen threshold may be applied by runtime to tails, but they were not optimized.",
                "heldout": "Index 110 is never read for calibration; evaluate only after policy.json has been frozen."},
        }
        return report, rounds_payload

    def write(self, output_directory):
        report, rounds_payload = self._analyze()
        policy = {key: report[key] for key in (
            "schema_version", "selected_threshold", "calibration_source_row_indices", "expected_max_new_tokens",
            "heldout_source_row_indices_excluded", "source_samples_path", "source_samples_sha256",
            "source_summary_path", "source_summary_sha256", "source_provenance", "selection", "interpretation", "rounds_dataset")}
        policy.update(status="frozen", calibration_report_sha256=_sha(_json_bytes(report)))
        output = Path(output_directory).resolve()
        files = {"calibration.json": _json_bytes(report), "policy.json": _json_bytes(policy), "rounds.jsonl": rounds_payload}
        if output.exists() and any(output.iterdir()):
            _require(all((output / name).is_file() and (output / name).read_bytes() == payload
                         for name, payload in files.items()),
                     "Frozen calibration output differs; refusing to overwrite policy or source evidence")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
            try:
                for name, payload in files.items():
                    with (temporary / name).open("xb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                os.rename(temporary, output)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        return {**report, "policy_path": str(output / "policy.json"),
                "calibration_path": str(output / "calibration.json"), "rounds_path": str(output / "rounds.jsonl"),
                "policy_sha256": _sha(files["policy.json"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Required complete output count per chain (default: 2048)")
    args = parser.parse_args(argv)
    result = ConfidenceCalibrator(args.samples, expected_max_new_tokens=args.max_new_tokens).write(args.output)
    print(f"Frozen threshold: {result['selected_threshold']:.2f}; policy: {result['policy_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
