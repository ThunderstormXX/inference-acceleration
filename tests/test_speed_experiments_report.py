"""Independent statistics, protocol separation and fail-closed claim eligibility."""
import gzip
import json
from pathlib import Path

import pytest

from inference_lab.optimizations.report import main, read_experiment, render_markdown


def measurement(speed, ids=None):
    return {"decode_tokens": 4, "decode_seconds": 4 / speed,
            "generated_token_ids": [1, 2, 3, 4, 5] if ids is None else ids}


def row(name, speed, repeat=0, phase="measurement", parity=True):
    return {"variant": name, "prompt_index": 0, "repeat": repeat, "phase": phase,
            "measurement": measurement(speed), "matches_baseline": parity,
            "sleep": {"sleep_detected": False}}


def save_kernel(path, *, parity=True, status="completed"):
    path.mkdir()
    runs = [row("ar", 10, -1, "reference"), row("ar", 10), row("fast", 20, parity=parity),
            row("ar", 20, 1), row("fast", 30, 1), row("ar", 10, 2, "bracket")]
    raw = {"status": status, "runs": runs, "errors": [],
           "protocol": {"variants": ["ar", "fast"], "repeats": 2, "prompts": [{}], "tokens": 5}}
    (path / "raw.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode()))
    (path / "summary.json").write_text(json.dumps({"status": status, "by_variant": {"ar": {}, "fast": {}}}))
    return raw


def test_kernel_means_ratios_and_counts_exclude_brackets(tmp_path):
    save_kernel(tmp_path / "kernel")
    result = read_experiment(tmp_path / "kernel")
    rows = {row["name"]: row for row in result["rows"]}
    assert rows["ar"]["tok_s"]["mean"] == 15
    assert rows["fast"]["tok_s"]["mean"] == 25
    assert rows["fast"]["tok_s"]["stdev"] == pytest.approx(50 ** .5)
    assert rows["fast"]["paired_speed_ratio"]["mean"] == 1.75
    assert rows["fast"]["measurement_count"] == 2
    assert rows["fast"]["unique_prompt_count"] == 1
    assert rows["fast"]["eligible_for_same_protocol_comparison"]
    assert result["ranking"] is None


def test_fast_parity_failure_is_visible_but_never_eligible(tmp_path):
    save_kernel(tmp_path / "bad", parity=False)
    result = read_experiment(tmp_path / "bad")
    fast = next(row for row in result["rows"] if row["name"] == "fast")
    assert fast["tok_s"]["mean"] == 25
    assert not fast["eligible_for_same_protocol_comparison"]
    assert "token parity failed" in fast["no_claim_reasons"]
    assert "diagnostic only" in render_markdown({"experiments": [result]})


def test_incomplete_error_and_missing_files_are_kept(tmp_path):
    save_kernel(tmp_path / "running", status="running")
    result = read_experiment(tmp_path / "running")
    assert not any(row["eligible_for_same_protocol_comparison"] for row in result["rows"])
    missing = read_experiment(tmp_path / "missing")
    assert missing["status"] == "unavailable" and missing["warnings"]
    assert main(["--inputs", str(tmp_path / "running"), str(tmp_path / "missing"),
                 "--output", str(tmp_path / "combined")]) == 0
    combined = json.loads((tmp_path / "combined.json").read_text())
    assert len(combined["experiments"]) == 2


def test_mtp_pairs_use_pair_ids_not_pooled_ar(tmp_path):
    path = tmp_path / "mtp"
    path.mkdir()
    runs = []
    for k, baseline, speculative in [(2, 10, 20), (3, 20, 10)]:
        for mode, speed in [("ar", baseline), ("mtp", speculative)]:
            runs.append({"mode": mode, "block_size": 1 if mode == "ar" else k,
                         "pair_id": str(k), "prompt_index": 0, "repeat": 0, "role": "paired",
                         "measurement": measurement(speed), "matches_baseline": True,
                         "sleep": {"sleep_detected": False}})
    raw = {"status": "completed", "runs": runs,
           "protocol": {"block_sizes": [2, 3], "prompts": [{}], "repeats": 1, "expected_runs": 4}}
    (path / "raw.json").write_text(json.dumps(raw))
    (path / "summary.json").write_text(json.dumps({"status": "completed", "by_mode": {}}))
    result = read_experiment(path)
    rows = {row["name"]: row for row in result["rows"]}
    assert rows["mtp_k2"]["paired_speed_ratio"]["mean"] == 2
    assert rows["mtp_k3"]["paired_speed_ratio"]["mean"] == .5
    assert rows["mtp_k2"]["tok_s"]["stdev"] is None
    assert rows["ar_k1"]["measurement_count"] == 2 and rows["ar_k1"]["unique_prompt_count"] == 1


def test_missing_raw_cannot_inherit_summary_claims(tmp_path):
    path = tmp_path / "summary-only"
    path.mkdir()
    (path / "summary.json").write_text(json.dumps({"status": "completed", "by_variant": {
        "fast": {"tok_s": {"n": 6, "mean": 999, "stdev": 0}, "all_token_ids_match": True}},
        "all_runs_awake": True}))
    result = read_experiment(path)["rows"][0]
    assert result["statistics_source"] == "summary_only"
    assert result["measurement_count"] is None
    assert not result["eligible_for_same_protocol_comparison"]
    assert result["all_token_ids_match"] is None


def test_sleep_mismatched_snapshot_and_missing_requested_runs_fail_closed(tmp_path):
    path = tmp_path / "partial"
    raw = save_kernel(path)
    raw["runs"].pop()
    raw["runs"][2]["sleep"]["sleep_detected"] = True
    (path / "raw.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode()))
    result = read_experiment(path)
    assert any("run count" in warning for warning in result["warnings"])
    assert not any(row["eligible_for_same_protocol_comparison"] for row in result["rows"])


def test_ngram_variants_use_role_and_neighbor_pairs(tmp_path):
    path = tmp_path / "ngram"
    path.mkdir()
    bracket = {"variant": "ar", "prompt_index": 0, "repeat": -1, "role": "bracket", "pair_id": None,
               "measurement": measurement(99), "matches_baseline": True, "sleep": {"sleep_detected": False}}
    runs = [bracket]
    for width, ar, speed in [(1, 10, 20), (2, 20, 10)]:
        for variant, value in [("ar", ar), (f"ngram-w{width}-m2", speed)]:
            runs.append({"variant": variant, "width": width, "min_match": 2,
                         "prompt_index": 0, "repeat": 0, "role": "paired", "pair_id": f"w{width}",
                         "measurement": measurement(value), "matches_baseline": True,
                         "sleep": {"sleep_detected": False}})
    runs.append({**bracket, "repeat": 1})
    raw = {"status": "completed", "runs": runs,
           "protocol": {"widths": [1, 2], "min_matches": [2], "prompts": [{}], "repeats": 1}}
    (path / "raw.json").write_text(json.dumps(raw))
    (path / "summary.json").write_text(json.dumps({"status": "completed", "by_variant": {}}))
    result = read_experiment(path)
    assert result["family"] == "ngram_sweep"
    assert result["expected_raw_runs"] == result["completed_raw_runs"] == 6
    rows = {row["name"]: row for row in result["rows"]}
    assert rows["ar"]["tok_s"]["mean"] == 15
    assert rows["ar"]["measurement_count"] == 2
    assert rows["ar"]["unique_prompt_count"] == 1
    assert rows["ngram-w1-m2"]["paired_speed_ratio"]["mean"] == 2
    assert rows["ngram-w2-m2"]["paired_speed_ratio"]["mean"] == .5
    assert all(row["eligible_for_same_protocol_comparison"] for row in rows.values())
    # A live snapshot must remain diagnostic, even if completed pairs exist.
    raw["status"] = "running"
    raw["runs"].pop()
    (path / "raw.json").write_text(json.dumps(raw))
    active = read_experiment(path)
    assert active["status"] == "running"
    assert not any(row["eligible_for_same_protocol_comparison"] for row in active["rows"])


def test_large_ar_bracket_drift_marks_completed_data_diagnostic(tmp_path):
    path = tmp_path / "drift"
    raw = save_kernel(path)
    raw["runs"][-1]["measurement"] = measurement(12)
    (path / "raw.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode()))
    result = read_experiment(path)
    assert result["status"] == "completed"
    assert result["drift_bracket"]["before_tok_s"] == 10
    assert result["drift_bracket"]["after_tok_s"] == 12
    assert result["drift_bracket"]["after_over_before"] == 1.2
    assert "exceeds 5%" in result["diagnostic_flags"][0]
    assert all(row["tok_s"] for row in result["rows"])
    assert not any(row["eligible_for_same_protocol_comparison"] for row in result["rows"])


def test_mtp_shortlist_stock_controls_and_direct_comparison(tmp_path):
    path = tmp_path / "shortlist"
    path.mkdir()
    bracket = {"mode": "ar", "block_size": 1, "prompt_index": 0, "repeat": -1,
               "role": "bracket", "pair_id": None, "measurement": measurement(10),
               "matches_baseline": True, "sleep": {"sleep_detected": False}}
    runs = [bracket]
    for repeat in (0, 1):
        for mode, ar_speed, method_speed in [("mtp-stock", 10, 20), ("mtp", 20, 30)]:
            pair_id = f"r{repeat}-{mode}"
            for kind, speed in [("ar", ar_speed), (mode, method_speed)]:
                runs.append({"mode": kind, "block_size": 1 if kind == "ar" else 3,
                             "prompt_index": 0, "repeat": repeat, "role": "paired", "pair_id": pair_id,
                             "measurement": measurement(speed), "matches_baseline": True,
                             "sleep": {"sleep_detected": False}})
    runs.append({**bracket, "repeat": 2})
    raw = {"status": "completed", "runs": runs,
           "protocol": {"block_sizes": [3], "prompts": [{}], "repeats": 2,
                        "expected_runs": 10, "compare_stock_draft": True}}
    declared = {"n": 2, "mean": 1.5, "stdev": 0}
    summary = {"status": "completed", "by_mode": {}, "shortlist_over_stock": declared,
               "draft_comparisons": [{"repeat": 0, "shortlist_over_stock": 1.5}]}
    (path / "raw.json").write_text(json.dumps(raw))
    (path / "summary.json").write_text(json.dumps(summary))
    result = read_experiment(path)
    rows = {row["name"]: row for row in result["rows"]}
    assert rows["ar_k1"]["measurement_count"] == 4
    assert rows["ar_k1"]["unique_prompt_count"] == 1
    assert rows["ar_k1"]["warnings"] == []
    assert rows["mtp-stock_k3"]["paired_speed_ratio"]["mean"] == 2
    assert rows["mtp_k3"]["paired_speed_ratio"]["mean"] == 1.5
    assert result["shortlist_over_stock"]["mean"] == 1.5
    assert result["shortlist_over_stock"]["n"] == 2
    assert result["reported_shortlist_over_stock"] == declared
    assert result["reported_draft_comparisons"] == summary["draft_comparisons"]
    direct = result["draft_comparison_by_block_size"]["3"]
    assert direct["eligible_for_same_protocol_comparison"]
    assert direct["unique_prompt_count"] == 1 and direct["paired_measurements"] == 2
    assert "Shortlist / stock ratio" in render_markdown({"experiments": [result]})
    # A failed shortlist trajectory remains visible but cannot win against stock.
    failed = next(row for row in raw["runs"] if row["mode"] == "mtp")
    failed["matches_baseline"] = False
    failed["measurement"]["generated_token_ids"] = [9] * 5
    (path / "raw.json").write_text(json.dumps(raw))
    result = read_experiment(path)
    direct = result["draft_comparison_by_block_size"]["3"]
    assert direct["all_token_ids_match"] is False
    assert not direct["eligible_for_same_protocol_comparison"]
