"""Independent arithmetic and fail-closed provenance checks for CPU calibration."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from inference_lab.experiments.confidence.calibration import (
    ConfidenceCalibrator, THRESHOLDS, _classification, _reliability, _dispersion, main,
)


def observation(score, accepted):
    return {"score": score, "p1": score, "p2": 1., "accepted_count": accepted}


def make_row(index, specifications=((.1, 0), (.4, 1), (.8, 2)), *, tail=False):
    generated = [1]
    events = [{"type": "commit", "t": .05, "token_ids": [1], "output_count": 1, "round": 0}]
    rounds = []
    specs = [(score, accepted, 2) for score, accepted in specifications]
    if tail:
        specs.append((.2, 0, 1))
    for number, (score, accepted, proposal_count) in enumerate(specs, 1):
        proposals = [number * 10 + i for i in range(proposal_count)]
        emitted = proposals[:accepted] + [number * 10 + 9]
        before = len(generated)
        generated.extend(emitted)
        events.append({"type": "draft", "t": float(number), "round": number,
                       "token_ids": list(proposals), "output_count": before})
        events.append({"type": "commit", "t": number + .3, "round": number, "token_ids": list(emitted),
                       "output_count": len(generated), "accepted_count": accepted, "draft_count": proposal_count,
                       "proposed_token_ids": list(proposals), "rejected_token_ids": proposals[accepted:],
                       "verification_completed_t": number + .1, "cache_commit_enqueued_t": number + .2})
        rounds.append({"round": number, "output_count_before": before,
                       "proposal_token_ids": list(proposals), "emitted_token_ids": list(emitted),
                       "proposal_probabilities": [score] + [1.] * (proposal_count - 1),
                       "confidence_product": score, "accepted_count": accepted,
                       "decision": "verify", "verified_draft_count": proposal_count})
    duration = float(len(specs) + 1)
    return {"source_row_index": index, "method": "confidence_collect",
            "generated_token_ids": generated, "generated_tokens": len(generated),
            "prefill_seconds": .1, "decode_seconds": duration - .1,
            "host_observation": {"valid": True}, "confidence_rounds": rounds,
            "generation_trace": {"schema_version": 1, "token_ids": list(generated), "events": events,
                                 "prefill_seconds": .1, "decode_seconds": duration - .1, "total_seconds": duration,
                                 "eos_token_ids": [], "instrumentation": {"enabled": True}}}


def save_collect(directory, rows=None):
    directory.mkdir(parents=True, exist_ok=True)
    rows = rows or [make_row(index) for index in range(100, 110)]
    samples = directory / "samples.jsonl"
    payload = "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows).encode()
    samples.write_bytes(payload)
    summary = {"status": "completed", "config": {"stage": "collect", "max_new_tokens": len(rows[0]["generated_token_ids"])},
               "methods": ["confidence_collect"], "completed_measurements": len(rows),
               "source_row_indices": sorted(row["source_row_index"] for row in rows),
               "samples_sha256": hashlib.sha256(payload).hexdigest(),
               "sources": {"data_manifest_sha256": "a" * 64, "model_manifest_sha256": "b" * 64,
                           "draft_manifest_sha256": "c" * 64, "policy_sha256": None}}
    (directory / "summary.json").write_text(json.dumps(summary))
    return samples


def analyze(samples, **kwargs):
    return ConfidenceCalibrator(samples, expected_max_new_tokens=7, **kwargs).analyze()


def test_confusion_counts_are_hand_computed():
    records = [observation(.1, 0), observation(.8, 2), observation(.9, 1), observation(.2, 2)]
    result = _classification(records, .5)
    assert result["confusion"] == dict(true_positive=1, false_positive=1, true_negative=1, false_negative=1)
    assert result["balanced_accuracy"] == result["accuracy"] == result["coverage"] == .5
    assert result["precision"] == result["recall"] == result["specificity"] == .5
    assert result["selected_rejected_first_rounds"] == 0
    assert result["selected_rejected_second_rounds"] == 1
    assert result["selected_rejected_second_fraction"] == .5


def test_no_selection_and_single_class_have_explicit_denominators():
    positive = _classification([observation(.2, 2), observation(.8, 2)], .5)
    assert positive["classes_present"] == 1 and positive["balanced_accuracy"] == .5
    assert positive["specificity"] is None
    negative = _classification([observation(.2, 0), observation(.8, 1)], 1.)
    assert negative["balanced_accuracy"] == 1 and negative["recall"] is None
    assert negative["precision"] is None and negative["selected_rejected_first_fraction"] is None


def test_grid_is_fixed_and_macro_weighting_is_not_round_weighting():
    # Chain A prefers low threshold (one positive); B prefers > .2 (nine negatives).
    # Each chain has equal weight: all grid thresholds tie at .5, choose zero.
    chains = {100: [observation(.1, 2)], 101: [observation(.2, 0)] * 9}
    grid, best = ConfidenceCalibrator._grid(chains)
    assert THRESHOLDS == tuple(i / 20 for i in range(21))
    assert len(grid) == 21
    assert best["threshold"] == 0 and best["macro_balanced_accuracy"] == .5
    assert grid[5]["macro_balanced_accuracy"] == .5
    assert _classification(chains[100] + chains[101], .25)["accuracy"] == .9
    assert grid[3]["macro_balanced_accuracy"] == 0.


def test_probability_reliability_has_independent_brier_and_bin_boundaries():
    stats = _reliability([(0., 0), (.1, 1), (.9, 1), (1., 0)])
    assert stats["brier_score"] == pytest.approx((0 + .81 + .01 + 1) / 4)
    assert [item["count"] for item in stats["bins"]] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    assert stats["bins"][9]["mean_score"] == .95
    assert stats["bins"][9]["observed_acceptance_rate"] == .5
    assert stats["expected_calibration_error_10_bins"] == pytest.approx((0 + .9 + 2 * .45) / 4)
    assert _reliability([])["brier_score"] is None
    assert _dispersion([1., 3.])["sample_variance"] == 2.
    assert _dispersion([1.])["sample_stddev"] is None


def test_complete_ten_chains_select_lowest_perfect_threshold_with_loco(tmp_path):
    samples = save_collect(tmp_path / "collect")
    result = analyze(samples)
    assert result["selected_threshold"] == .45
    assert result["selection"]["macro_balanced_accuracy"] == 1.
    assert result["calibration_source_row_indices"] == list(range(100, 110))
    assert result["eligible_rounds"] == 30 and result["excluded_one_proposal_tail_rounds"] == 0
    acceptance = result["observed_acceptance"]
    assert acceptance["first_acceptance_rate"] == pytest.approx(2 / 3)
    assert acceptance["second_acceptance_given_first_rate"] == .5
    assert acceptance["both_acceptance_rate"] == pytest.approx(1 / 3)
    assert acceptance["rejected_first_rounds"] == acceptance["rejected_second_rounds"] == 10
    reliability = result["score_reliability"]
    assert reliability["product_vs_both_accepted"]["brier_score"] == pytest.approx((.01 + .16 + .04) / 3)
    assert reliability["p1_vs_first_accepted"]["brier_score"] == pytest.approx((.01 + .36 + .04) / 3)
    assert reliability["p2_vs_second_accepted_given_first"]["count"] == 20
    assert reliability["p2_vs_second_accepted_given_first"]["brier_score"] == .5
    folds = result["leave_one_chain_out"]["folds"]
    assert len(folds) == 10
    for fold in folds:
        assert len(fold["training_source_row_indices"]) == 9
        assert fold["left_out_source_row_index"] not in fold["training_source_row_indices"]
        assert fold["selected_threshold"] == .45 and fold["left_out_metrics"]["balanced_accuracy"] == 1.
    assert result["leave_one_chain_out"]["selected_threshold_dispersion"]["sample_variance"] == 0
    assert result["source_samples_sha256"] == hashlib.sha256(samples.read_bytes()).hexdigest()


def test_loco_chooses_threshold_without_left_out_chain():
    # A optimum .25; B optimum .65. Removing either selects only the other's optimum.
    chains = {100: [observation(.2, 0), observation(.4, 2)],
              101: [observation(.6, 0), observation(.8, 2)]}
    assert ConfidenceCalibrator._grid({100: chains[100]})[1]["threshold"] == .25
    assert ConfidenceCalibrator._grid({101: chains[101]})[1]["threshold"] == .65
    assert _classification(chains[100], .65)["balanced_accuracy"] == .5
    assert _classification(chains[101], .25)["balanced_accuracy"] == .5


def test_final_one_proposal_tail_is_excluded_from_all_selection_statistics(tmp_path):
    rows = [make_row(index, tail=True) for index in range(100, 110)]
    samples = save_collect(tmp_path / "collect", rows)
    result = ConfidenceCalibrator(samples, expected_max_new_tokens=8).analyze()
    assert result["selected_threshold"] == .45
    assert result["eligible_rounds"] == 30 and result["excluded_one_proposal_tail_rounds"] == 10
    assert all(chain["excluded_one_proposal_tail_rounds"] == 1 for chain in result["per_chain"])


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows[0].update(source_row_index=110),
    lambda rows: rows[0].update(source_row_index=101),
    lambda rows: rows[0].update(source_row_index=True),
    lambda rows: rows[0].update(method="confidence_gate"),
    lambda rows: rows[0]["generated_token_ids"].pop(),
    lambda rows: rows[0]["generation_trace"]["events"].pop(),
    lambda rows: rows[0].update(confidence_rounds=[]),
    lambda rows: rows[0]["confidence_rounds"].pop(),
    lambda rows: rows[0]["confidence_rounds"][0].update(decision="fallback"),
    lambda rows: rows[0]["confidence_rounds"][0].update(accepted_count=True),
    lambda rows: rows[0]["confidence_rounds"][0].update(accepted_count=3),
    lambda rows: rows[0]["confidence_rounds"][0].update(accepted_count=2),
    lambda rows: rows[0]["confidence_rounds"][0].update(output_count_before=2),
    lambda rows: rows[0]["confidence_rounds"][0].update(confidence_product=.9),
    lambda rows: rows[0]["confidence_rounds"][0].update(confidence_product=float("nan")),
    lambda rows: rows[0]["confidence_rounds"][0].update(proposal_probabilities=[True, 1]),
    lambda rows: rows[0]["confidence_rounds"][0].update(proposal_probabilities=[-1, 1]),
    lambda rows: rows[0]["confidence_rounds"][0].update(proposal_probabilities=[1, float("inf")]),
    lambda rows: rows[0]["confidence_rounds"][0].update(proposal_token_ids=[999, 998]),
    lambda rows: rows[0]["confidence_rounds"][0].update(emitted_token_ids=[999]),
    lambda rows: rows[0]["confidence_rounds"][0].update(proposal_probabilities=[]),
    lambda rows: rows[0]["confidence_rounds"][0].update(proposal_probabilities=[.1, 1, 1]),
    lambda rows: rows[0]["host_observation"].update(valid=False),
])
def test_rejects_partial_leaking_or_inconsistent_collect_rows(tmp_path, mutation):
    rows = [make_row(index) for index in range(100, 110)]
    mutation(rows)
    samples = save_collect(tmp_path / "collect", rows)
    with pytest.raises((ValueError, TypeError)):
        analyze(samples)


@pytest.mark.parametrize("mutation", [
    lambda summary: summary.update(status="running"),
    lambda summary: summary.update(status="failed"),
    lambda summary: summary["config"].update(stage="smoke"),
    lambda summary: summary["config"].update(max_new_tokens=2048),
    lambda summary: summary.update(methods=["confidence_collect", "stock_mtp"]),
    lambda summary: summary.update(samples_sha256="f" * 64),
    lambda summary: summary.update(completed_measurements=9),
    lambda summary: summary.update(source_row_indices=list(range(101, 111))),
    lambda summary: summary["sources"].update(data_manifest_sha256=None),
    lambda summary: summary["sources"].update(model_manifest_sha256="bad"),
    lambda summary: summary["sources"].update(policy_sha256="d" * 64),
])
def test_rejects_incomplete_or_changed_summary(tmp_path, mutation):
    samples = save_collect(tmp_path / "collect")
    path = samples.parent / "summary.json"
    summary = json.loads(path.read_text())
    mutation(summary)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        analyze(samples)


def test_default_requires_2048_complete_output_tokens(tmp_path):
    samples = save_collect(tmp_path / "collect")
    with pytest.raises(ValueError, match="budget"):
        ConfidenceCalibrator(samples).analyze()


@pytest.mark.parametrize("indices", [[], [100, 100], [True], [-1], [110], [100, 110]])
def test_expected_indices_cannot_include_heldout_or_duplicates(tmp_path, indices):
    with pytest.raises(ValueError):
        ConfidenceCalibrator(tmp_path / "samples.jsonl", indices)


def test_policy_is_frozen_and_identical_rerun_does_not_write(tmp_path):
    samples = save_collect(tmp_path / "collect")
    calibrator = ConfidenceCalibrator(samples, expected_max_new_tokens=7)
    output = tmp_path / "policy"
    written = calibrator.write(output)
    policy_path = output / "policy.json"
    before = {name: ((output / name).stat().st_mtime_ns, (output / name).read_bytes())
              for name in ("policy.json", "calibration.json", "rounds.jsonl")}
    policy = json.loads(policy_path.read_bytes())
    assert policy["status"] == "frozen" and policy["selected_threshold"] == .45
    assert policy["source_samples_sha256"] == hashlib.sha256(samples.read_bytes()).hexdigest()
    assert policy["calibration_report_sha256"] == hashlib.sha256((output / "calibration.json").read_bytes()).hexdigest()
    assert written["policy_sha256"] == hashlib.sha256(policy_path.read_bytes()).hexdigest()
    assert calibrator.write(output) == written
    assert before == {name: ((output / name).stat().st_mtime_ns, (output / name).read_bytes()) for name in before}
    rows = [make_row(index, specifications=((.2, 0), (.4, 1), (.9, 2))) for index in range(100, 110)]
    save_collect(samples.parent, rows)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        calibrator.write(output)
    assert policy_path.read_bytes() == before["policy.json"][1]


def test_tampered_frozen_report_is_not_silently_repaired(tmp_path):
    samples = save_collect(tmp_path / "collect")
    calibrator = ConfidenceCalibrator(samples, expected_max_new_tokens=7)
    output = tmp_path / "policy"
    calibrator.write(output)
    (output / "calibration.json").write_text("{}")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        calibrator.write(output)


def test_cli_writes_the_declared_artifacts(tmp_path, capsys):
    samples = save_collect(tmp_path / "collect")
    output = tmp_path / "policy"
    output.mkdir()  # Caller may create an empty directory before freezing.
    assert main(["--samples", str(samples), "--output", str(output), "--max-new-tokens", "7"]) == 0
    assert "Frozen threshold: 0.45" in capsys.readouterr().out
    assert sorted(path.name for path in output.iterdir()) == ["calibration.json", "policy.json", "rounds.jsonl"]


def test_loco_report_trains_on_other_chains_and_reports_sample_variance(tmp_path):
    rows = [make_row(100, ((.2, 0), (.3, 1), (.4, 2))),
            make_row(101, ((.6, 0), (.7, 1), (.8, 2)))]
    result = analyze(save_collect(tmp_path / "collect", rows), expected_indices=[100, 101])
    assert result["selected_threshold"] == .35
    assert result["selection"]["macro_balanced_accuracy"] == .75
    folds = result["leave_one_chain_out"]["folds"]
    assert [fold["selected_threshold"] for fold in folds] == [.75, .35]
    assert [fold["training_source_row_indices"] for fold in folds] == [[101], [100]]
    assert all(fold["left_out_metrics"]["balanced_accuracy"] == .5 for fold in folds)
    assert result["leave_one_chain_out"]["selected_threshold_dispersion"]["sample_variance"] == pytest.approx(.08)
    assert result["per_chain_dispersion"]["balanced_accuracy_at_selected_threshold"]["sample_variance"] == .125


def test_optional_partial_round_is_not_allowed_before_budget_tail(tmp_path):
    rows = [make_row(index) for index in range(100, 110)]
    row = rows[0]
    first = row["confidence_rounds"][0]
    first["proposal_probabilities"].pop()
    first["proposal_token_ids"].pop()
    first["verified_draft_count"] = 1
    draft, commit = row["generation_trace"]["events"][1:3]
    draft["token_ids"].pop()
    commit["proposed_token_ids"].pop()
    commit["rejected_token_ids"].pop()
    commit["draft_count"] = 1
    with pytest.raises(ValueError, match="shortened proposal"):
        analyze(save_collect(tmp_path / "collect", rows))


def test_calibration_does_not_import_gpu_packages(tmp_path):
    import os
    import subprocess
    import sys
    code = ('import sys; from inference_lab.experiments.confidence.calibration import ConfidenceCalibrator; '
            'assert not any(n == "mlx" or n.startswith("mlx.") or n == "torch" or n.startswith("torch.") '
            'for n in sys.modules)')
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert result.returncode == 0, result.stderr


def test_flat_rounds_export_preserves_context_labels_and_exact_source_ids(tmp_path):
    rows = [make_row(index) for index in range(100, 110)]
    for row in rows:
        for record, commit in zip(row["confidence_rounds"], row["generation_trace"]["events"][2::2]):
            record["target_token_ids"] = [70, 71, 72]
            commit["target_token_ids"] = [70, 71, 72]
    samples = save_collect(tmp_path / "collect", rows)
    output = tmp_path / "policy"
    result = ConfidenceCalibrator(samples, expected_max_new_tokens=7).write(output)
    payload = (output / "rounds.jsonl").read_bytes()
    flat = [json.loads(line) for line in payload.splitlines()]
    assert len(flat) == 30
    assert sorted({row["source_row_index"] for row in flat}) == list(range(100, 110))
    assert all(row["source_row_index"] != 110 and len(row["proposal_token_ids"]) == 2 for row in flat)
    assert [row["second_accepted_given_first"] for row in flat[:3]] == [None, False, True]
    assert [row["first_accepted"] for row in flat[:3]] == [False, True, True]
    assert [row["both_accepted"] for row in flat[:3]] == [False, False, True]
    assert [row["accepted_count"] for row in flat[:3]] == [0, 1, 2]
    for exported, record in zip(flat[:3], rows[0]["confidence_rounds"]):
        before = record["output_count_before"]
        assert exported["round"] == record["round"]
        assert exported["output_count_before"] == before
        assert exported["previous_generated_token_ids"] == rows[0]["generated_token_ids"][max(0, before-12):before]
        assert exported["proposal_token_ids"] == record["proposal_token_ids"]
        assert exported["emitted_token_ids"] == record["emitted_token_ids"]
        assert exported["target_token_ids"] == [70, 71, 72]
        assert exported["p1"] * exported["p2"] == exported["confidence_product"]
    expected_metadata = {"schema_version": 1, "filename": "rounds.jsonl", "count": 30,
                         "sha256": hashlib.sha256(payload).hexdigest(), "unit": "verified_full_two_proposal_round"}
    assert result["rounds_dataset"] == expected_metadata
    for name in ("calibration.json", "policy.json"):
        assert json.loads((output / name).read_bytes())["rounds_dataset"] == expected_metadata
    assert result["selected_threshold"] == .45


def test_flat_rounds_excludes_tail_and_uses_twelve_ids_without_tokenizer(tmp_path):
    rows = [make_row(index, specifications=((.2, 0),) * 15 + ((.9, 2),), tail=True)
            for index in range(100, 110)]
    samples = save_collect(tmp_path / "collect", rows)
    budget = len(rows[0]["generated_token_ids"])
    result = ConfidenceCalibrator(samples, expected_max_new_tokens=budget).write(tmp_path / "policy")
    flat = [json.loads(line) for line in Path(result["rounds_path"]).read_bytes().splitlines()]
    assert len(flat) == 160
    last = [row for row in flat if row["source_row_index"] == 100][-1]
    before = last["output_count_before"]
    assert before == 16 and len(last["previous_generated_token_ids"]) == 12
    assert last["previous_generated_token_ids"] == rows[0]["generated_token_ids"][4:16]
    assert last["target_token_ids"] is None  # No invented verifier output when unavailable.
    assert all(len(row["proposal_token_ids"]) == 2 for row in flat)
    assert result["excluded_one_proposal_tail_rounds"] == 10


def test_flat_rounds_are_part_of_the_immutable_frozen_bundle(tmp_path):
    samples = save_collect(tmp_path / "collect")
    calibrator = ConfidenceCalibrator(samples, expected_max_new_tokens=7)
    output = tmp_path / "policy"
    calibrator.write(output)
    original_policy = (output / "policy.json").read_bytes()
    (output / "rounds.jsonl").write_text('{}\n')
    with pytest.raises(ValueError, match="refusing to overwrite"):
        calibrator.write(output)
    assert (output / "policy.json").read_bytes() == original_policy
