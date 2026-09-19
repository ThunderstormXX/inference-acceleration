"""CPU-only validation of explicit, immutable speculative benchmark evidence."""
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from inference_lab.benchmarking.speculative_report import SpeculativeReport, main
from inference_lab.core.config import BenchmarkConfig


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def write_rows(directory, rows):
    (directory / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def mutate_summary(directory, update):
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    update(summary)
    path.write_text(json.dumps(summary))


def make_run(root, label, speculative=False, factor=1, count=2, family="dflash"):
    directory = root / label
    directory.mkdir(parents=True)
    config = asdict(BenchmarkConfig("mlx-dflash" if speculative else "mlx", count=count,
                                   max_new_tokens=6, label=label, wired_memory=True))
    if family == "mtp":
        config["backend"] = "mlx-mtp" if speculative else "mlx-vlm"
    if speculative:
        config.update(block_size=5, draft_bits=4, draft_path="/fixture/draft")
    prompts = [{"index": i, "prompt_tokens": [1000 + i, 42]} for i in range(count)]
    rows = []
    for i, prompt in enumerate(prompts):
        rate = 10 * (i + 1) * factor
        row = {"index": i, "prompt_token_sha256": digest(prompt["prompt_tokens"]),
               "prompt_tokens": 2, "generated_tokens": 6, "decode_tokens": 5,
               "prefill_seconds": 2 / (rate * 5), "decode_seconds": 5 / rate,
               "generated_token_ids": list(range(6)), "timing_method": "synchronized; first token in prefill"}
        if speculative:
            row.update(drafted_tokens=4 * (i + 1), accepted_draft_tokens=2 + 4 * i,
                       speculative_rounds=3, emitted_draft_tokens=2 + 2 * i,
                       emitted_target_tokens=4 - 2 * i,
                       draft_acceptance_rate=-999)  # Recompute from raw counts.
        rows.append(row)
    summary = {"status": "completed", "config": config, "completed_samples": count,
               "environment": {"caffeinate_assertions": "di", "packages": {"mlx": "0.32.2", "mlx-lm": "0.31.3"}},
               "backend": {"sampling": "greedy", "wired_memory": True,
                           "versions": {"mlx": "0.32.2", "mlx-lm": "0.31.3"}},
               "model_manifest_sha256": "a" * 64, "dataset_sha256": "b" * 64,
               "prompt_tokens_sha256": digest([p["prompt_tokens"] for p in prompts]),
               "metrics": {"decode": {"mean_tokens_per_second": -999}}}
    if family == "mtp":
        summary["environment"]["packages"]["mlx-vlm"] = "0.7.1"
        summary["backend"]["versions"]["mlx-vlm"] = "0.7.1"
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "prompts.json").write_text(json.dumps(prompts))
    write_rows(directory, rows)
    return directory


@pytest.fixture
def evidence(tmp_path):
    baseline = make_run(tmp_path / "runs", "spec-baseline")
    speculative = make_run(tmp_path / "runs", "dflash-k5-q4", True, factor=2)
    return SpeculativeReport(baseline, [speculative], tmp_path / "reports")


def test_recomputes_raw_stats_weighted_acceptance_and_preserves_files(evidence):
    files = {path: path.read_bytes() for directory in [evidence.baseline, *evidence.speculative] for path in directory.iterdir()}
    report = evidence.build()
    assert report["status"] == "completed" and report["parity_status"] == "passed"
    comparison = report["comparisons"][0]
    assert report["baseline"]["statistics"]["decode"]["mean_tokens_per_second"] == 15
    assert comparison["statistics"]["decode"]["mean_tokens_per_second"] == 30
    assert comparison["statistics"]["decode"]["sample_stddev_tokens_per_second"] == pytest.approx(200 ** .5)
    assert comparison["statistics"]["decode"]["sample_variance_tokens_per_second"] == 200
    assert comparison["speedup"]["decode"] == {"ratio_of_mean_rates": 2, "ratio_of_aggregate_rates": 2}
    assert comparison["speculation"]["weighted_draft_acceptance_rate"] == pytest.approx(8 / 12)
    assert comparison["speculation"]["mean_committed_tokens_per_round"] == pytest.approx(10 / 6)
    for path, content in files.items():
        assert path.read_bytes() == content
    for source in report["sources"].values():
        for filename, item in source["files"].items():
            assert item["sha256"] == hashlib.sha256(files[Path(item["path"])]).hexdigest()
    records = list(csv.DictReader((evidence.output / "speculative-comparison.csv").open()))
    assert len(records) == 4
    assert all(row["summary_sha256"] and row["baseline_summary_sha256"] for row in records)
    markdown = (evidence.output / "speculative-comparison.md").read_text()
    assert "±" in markdown and "ddof=1" in markdown
    assert "не является доверительным интервалом" in markdown


def test_output_mismatch_is_usable_but_prominently_qualified(evidence):
    directory = evidence.speculative[0]
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    rows[1]["generated_token_ids"][2] = 777
    write_rows(directory, rows)
    report = evidence.build()
    assert report["status"] == "completed" and report["parity_status"] == "failed"
    parity = report["comparisons"][0]["parity"]
    assert parity["matched_prompts"] == 1
    assert parity["per_prompt"][1] == {"index": 1, "matched": False, "matching_prefix_tokens": 2,
                                       "first_mismatch_token_position": 2, "baseline_token_id": 2, "speculative_token_id": 777}
    text = (evidence.output / "speculative-comparison.md").read_text()
    assert "не пройдена" in text and "baseline ID 2" in text and "speculative ID 777" in text
    assert "точная эквивалентность baseline не подтверждена" in text
    main(["--baseline", str(evidence.baseline), "--speculative", str(directory), "--output", str(evidence.output)])


@pytest.mark.parametrize("fault", ["partial", "count", "max_new_tokens", "dataset", "model", "assertion", "wired", "backend_wired", "label", "version", "sampling"])
def test_rejects_incompatible_summary(evidence, fault):
    def change(summary):
        if fault == "partial":
            summary["status"] = "running"
        elif fault in ("count", "max_new_tokens"):
            summary["config"][fault] += 1
        elif fault in ("dataset", "model"):
            summary["dataset_sha256" if fault == "dataset" else "model_manifest_sha256"] = "c" * 64
        elif fault == "assertion":
            summary["environment"]["caffeinate_assertions"] = "i"
        elif fault == "wired":
            summary["config"]["wired_memory"] = False
        elif fault == "backend_wired":
            summary["backend"]["wired_memory"] = False
        elif fault == "label":
            summary["config"]["label"] = "historical-awake"
        elif fault == "version":
            summary["environment"]["packages"]["mlx"] = "0.31.0"
            summary["backend"]["versions"]["mlx"] = "0.31.0"
        else:
            summary["backend"]["sampling"] = "temperature=0.8"
    mutate_summary(evidence.baseline if fault == "label" else evidence.speculative[0], change)
    report = evidence.build()
    assert report["status"] == "failed" and not report["comparisons"] and report["errors"]
    assert all(source["files"]["summary.json"]["sha256"] for source in report["sources"].values())
    assert "Сравнение не выполнено" in (evidence.output / "speculative-comparison.md").read_text()
    with pytest.raises(SystemExit) as error:
        main(["--baseline", str(evidence.baseline), "--speculative", str(evidence.speculative[0]), "--output", str(evidence.output)])
    assert error.value.code == 1


def test_different_inputs_rejected_even_if_individual_hashes_are_valid(evidence):
    directory = evidence.speculative[0]
    prompts = json.loads((directory / "prompts.json").read_text())
    prompts[0]["prompt_tokens"][0] = 444
    (directory / "prompts.json").write_text(json.dumps(prompts))
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    rows[0]["prompt_token_sha256"] = digest(prompts[0]["prompt_tokens"])
    write_rows(directory, rows)
    mutate_summary(directory, lambda summary: summary.update(prompt_tokens_sha256=digest([p["prompt_tokens"] for p in prompts])))
    report = evidence.analyze()
    assert report["status"] == "failed"
    assert "exact input prompt indices/token IDs" in str(report["errors"])


@pytest.mark.parametrize("fault", ["missing", "row_hash", "row_index", "truncated", "counter", "duration", "generated_ids"])
def test_corrupt_raw_evidence_rejected(evidence, fault):
    directory = evidence.speculative[0]
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    if fault == "missing":
        (directory / "prompts.json").unlink()
    elif fault == "truncated":
        rows.pop()
    elif fault == "row_hash":
        rows[0]["prompt_token_sha256"] = "bad"
    elif fault == "row_index":
        rows[0]["index"] = 999
    elif fault == "counter":
        rows[0]["accepted_draft_tokens"] = 999
    elif fault == "duration":
        rows[0]["decode_seconds"] = float("nan")
    else:
        rows[0]["generated_token_ids"].pop()
    write_rows(directory, rows)
    report = evidence.analyze()
    assert report["status"] == "failed" and report["comparisons"] == []


def test_single_sample_has_undefined_dispersion_and_multiple_candidates(tmp_path):
    baseline = make_run(tmp_path / "runs", "spec-baseline-short", count=1)
    candidates = [make_run(tmp_path / "runs", f"dflash-k{k}", True, factor=k, count=1) for k in (3, 5)]
    builder = SpeculativeReport(baseline, candidates, tmp_path / "reports")
    report = builder.build()
    assert report["status"] == "completed" and len(report["comparisons"]) == 2
    for row in report["comparisons"]:
        assert row["statistics"]["samples"] == 1
        assert row["statistics"]["decode"]["sample_stddev_tokens_per_second"] is None
        assert row["statistics"]["decode"]["sample_variance_tokens_per_second"] is None
    assert "SD н/д" in (builder.output / "speculative-comparison.md").read_text()


def test_explicit_paths_no_candidates_or_repeated_paths_rejected(evidence):
    assert SpeculativeReport(evidence.baseline, []).analyze()["status"] == "failed"
    assert SpeculativeReport(evidence.baseline, [evidence.baseline]).analyze()["status"] == "failed"
    assert SpeculativeReport(evidence.baseline, evidence.speculative * 2).analyze()["status"] == "failed"


def test_missing_version_is_explicit_limitation_and_inconsistent_metadata_rejected(evidence):
    directory = evidence.speculative[0]
    mutate_summary(directory, lambda summary: summary["backend"].update(versions={"mlx": "9.0.0"}))
    report = evidence.analyze()
    assert report["status"] == "failed" and "Conflicting" in str(report["errors"])
    mutate_summary(directory, lambda summary: summary["backend"].update(versions={}))
    mutate_summary(directory, lambda summary: summary["environment"].update(packages={}))
    report = evidence.analyze()
    assert report["status"] == "completed" and any("mlx unavailable" in row["message"] for row in report["warnings"])


@pytest.mark.parametrize("suffix", ["", "-repeat2"])
def test_mtp_family_compares_with_its_own_baseline(tmp_path, suffix):
    baseline = make_run(tmp_path / "runs", "mtp-baseline" + suffix, family="mtp")
    candidate = make_run(tmp_path / "runs", "mtp-k2", speculative=True, family="mtp", factor=2)
    builder = SpeculativeReport(baseline, [candidate], tmp_path / "reports/mtp")
    report = builder.build()
    assert report["status"] == "completed" and report["parity_status"] == "passed"
    assert report["comparisons"][0]["speedup"]["decode"]["ratio_of_mean_rates"] == 2
    assert report["sources"][str(candidate)]["mlx_versions"]["mlx-vlm"] == "0.7.1"
    assert (builder.output / "speculative-comparison.json").is_file()


@pytest.mark.parametrize("baseline_family,candidate_family", [("dflash", "mtp"), ("mtp", "dflash")])
def test_cross_family_comparisons_rejected(tmp_path, baseline_family, candidate_family):
    label = "spec-baseline" if baseline_family == "dflash" else "mtp-baseline"
    baseline = make_run(tmp_path / "runs", label, family=baseline_family)
    candidate = make_run(tmp_path / "runs", "candidate", speculative=True, family=candidate_family)
    report = SpeculativeReport(baseline, [candidate]).analyze()
    assert report["status"] == "failed" and not report["comparisons"]
    assert "backend family" in str(report["errors"])


def test_mixed_family_candidate_list_is_not_partially_published(evidence):
    mtp = make_run(evidence.baseline.parent, "mtp-k2", speculative=True, family="mtp")
    report = SpeculativeReport(evidence.baseline, [*evidence.speculative, mtp]).analyze()
    assert report["status"] == "failed" and report["comparisons"] == []


@pytest.mark.parametrize("family,label", [("mtp", "spec-baseline"), ("dflash", "mtp-baseline")])
def test_family_specific_baseline_label_required(tmp_path, family, label):
    baseline = make_run(tmp_path / "runs", label, family=family)
    candidate = make_run(tmp_path / "runs", "candidate", speculative=True, family=family)
    report = SpeculativeReport(baseline, [candidate]).analyze()
    assert report["status"] == "failed" and "Baseline label" in str(report["errors"])


def test_mtp_rejects_different_mlx_vlm_versions(tmp_path):
    baseline = make_run(tmp_path / "runs", "mtp-baseline", family="mtp")
    candidate = make_run(tmp_path / "runs", "mtp-k2", speculative=True, family="mtp")
    def change(summary):
        summary["environment"]["packages"]["mlx-vlm"] = "0.7.2"
        summary["backend"]["versions"]["mlx-vlm"] = "0.7.2"
    mutate_summary(candidate, change)
    report = SpeculativeReport(baseline, [candidate]).analyze()
    assert report["status"] == "failed" and "package mlx-vlm" in str(report["errors"])


def test_dflash_does_not_require_unrelated_mlx_vlm_version_equality(evidence):
    for directory, version in ((evidence.baseline, "0.7.1"), (evidence.speculative[0], "0.7.2")):
        mutate_summary(directory, lambda summary: summary["environment"]["packages"].update({"mlx-vlm": version}))
    assert evidence.analyze()["status"] == "completed"
