"""CPU-only evidence checks for exact-prompt repeat statistics."""

import csv
import hashlib
import json
from dataclasses import asdict

import pytest

from inference_lab.benchmarking.recheck_report import RecheckReport
from inference_lab.core.config import BenchmarkConfig


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def make_run(runs, backend, cohort, stamp="20260101T000000Z", factor=None):
    label, count = RecheckReport.COHORTS[cohort]
    factor = factor if factor is not None else (2 if cohort == "recheck5" else 1)
    folder = runs / f"{stamp}-{backend}-{label}"
    folder.mkdir(parents=True)
    config = asdict(BenchmarkConfig(backend, count=count, label=label,
                                   wired_memory=backend in ("mlx", "mlx-vlm")))
    prompts = [{"index": i, "prompt_tokens": [1000 + i, 42]} for i in range(count)]
    rows = []
    for i, prompt in enumerate(prompts):
        # Full-run statistics intentionally differ greatly from the first five.
        rate = (10 * (i + 1) if i < 5 else 1000) * factor
        rows.append({"index": i, "prompt_token_sha256": digest(prompt["prompt_tokens"]),
                     "prompt_tokens": 2, "generated_tokens": 128, "decode_tokens": 127,
                     "prefill_seconds": 2 / rate, "decode_seconds": 127 / (rate / 5),
                     "generated_token_ids": list(range(128)), "timing_method": f"{backend} synchronized"})
    summary = {"status": "completed", "started_at": stamp, "config": config,
               "completed_samples": count, "environment": {"caffeinate_assertions": "di"},
               "model_manifest_sha256": "a" * 64, "dataset_sha256": "b" * 64,
               "prompt_tokens_sha256": digest([p["prompt_tokens"] for p in prompts]),
               # This stale value must never be used to derive the report.
               "metrics": {"decode": {"mean_tokens_per_second": -999}}}
    (folder / "summary.json").write_text(json.dumps(summary))
    (folder / "prompts.json").write_text(json.dumps(prompts))
    (folder / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return folder


@pytest.fixture
def evidence(tmp_path):
    runs = tmp_path / "runs"
    folders = {(backend, cohort): make_run(runs, backend, cohort)
               for backend in RecheckReport.BACKENDS for cohort in RecheckReport.COHORTS}
    return RecheckReport(runs, tmp_path / "reports"), folders


def rewrite_summary(folder, update):
    path = folder / "summary.json"
    summary = json.loads(path.read_text())
    update(summary)
    path.write_text(json.dumps(summary))


def test_recomputes_sample_statistics_and_compares_matching_first_five(evidence):
    builder, folders = evidence
    before = {path: path.read_bytes() for folder in folders.values() for path in folder.iterdir()}
    report = builder.build()
    assert report["status"] == "completed"
    assert report["errors"] == []
    for row in report["comparisons"]:
        assert row["old_first5"]["samples"] == row["recheck5"]["samples"] == 5
        assert row["awake100"]["samples"] == 100
        assert row["old_first5"]["prefill"]["mean_tokens_per_second"] == 30
        assert row["recheck5"]["prefill"]["sample_variance_tokens_per_second"] == pytest.approx(1000)
        assert row["recheck5"]["prefill"]["sample_stddev_tokens_per_second"] == pytest.approx(1000 ** .5)
        assert row["delta_percent"]["prefill"] == pytest.approx(100)
        assert row["delta_percent"]["decode"] == pytest.approx(100)
        assert row["awake100"]["prefill"]["mean_tokens_per_second"] > 900
    for path, data in before.items():
        assert path.read_bytes() == data
    saved = json.loads((builder.output / "recheck-5.json").read_text())
    assert saved["status"] == "completed"
    source = saved["sources"]["mlx"]["recheck5"]["files"]["samples.jsonl"]
    assert source["sha256"] == hashlib.sha256(before[folders["mlx", "recheck5"] / "samples.jsonl"]).hexdigest()
    records = list(csv.DictReader((builder.output / "recheck-5.csv").open()))
    assert len(records) == 4 * 3 * 2
    assert all(record["summary_sha256"] and record["prompts_sha256"] and record["samples_sha256"] for record in records)
    markdown = (builder.output / "recheck-5.md").read_text()
    assert "±" in markdown and "ddof=1" in markdown
    assert "не является доверительным интервалом" in markdown
    assert "не оценивает дисперсию повторных запусков одного prompt" in markdown


def test_latest_completed_selected_and_partial_never_used(evidence):
    builder, folders = evidence
    latest = make_run(builder.runs, "mlx", "recheck5", stamp="20260102T000000Z", factor=3)
    partial = make_run(builder.runs, "mlx", "recheck5", stamp="20260103T000000Z", factor=100)
    rewrite_summary(partial, lambda summary: summary.update(status="failed", completed_samples=2))
    report = builder.analyze()
    assert report["status"] == "completed"
    assert report["sources"]["mlx"]["recheck5"]["files"]["summary.json"]["path"] == str((latest / "summary.json").resolve())
    row = next(row for row in report["comparisons"] if row["backend"] == "mlx")
    assert row["recheck5"]["prefill"]["mean_tokens_per_second"] == 90
    assert any(item.get("status") == "failed" for item in report["excluded_runs"])


def test_power_conditions_are_visible_without_claiming_causal_framework_change(evidence):
    builder, folders = evidence
    for (backend, cohort), folder in folders.items():
        source = "AC Power" if cohort == "awake100" else "Battery Power"
        def update(summary):
            power = {"status": "available", "raw": f"Now drawing from '{source}'\n -InternalBattery-0\t98%; discharging; 2:00 remaining"}
            summary["environment"]["power_source"] = power
            summary["resources_after"] = {"power_source": power}
        rewrite_summary(folder, update)
    assert builder.build()["status"] == "completed"
    text = (builder.output / "recheck-5.md").read_text()
    assert "AC Power, 98%, discharging" in text
    assert "Battery Power, 98%, discharging" in text
    assert "причинный эффект батареи не измерен" in text


@pytest.mark.parametrize("field", ["dataset_sha256", "model_manifest_sha256", "max_new_tokens", "wired_memory"])
def test_rejects_hash_or_protocol_mismatch(evidence, field):
    builder, folders = evidence
    def update(summary):
        if field.endswith("sha256"):
            summary[field] = "c" * 64
        else:
            summary["config"][field] = 64 if field == "max_new_tokens" else False
    rewrite_summary(folders["mlx", "recheck5"], update)
    report = builder.build()
    assert report["status"] == "failed"
    assert not report["comparisons"]
    assert report["errors"]
    assert "±" not in (builder.output / "recheck-5.md").read_text()
    assert all(row["status"] == "failed" for row in csv.DictReader((builder.output / "recheck-5.csv").open()))


def test_rejects_different_prompt_ids_even_with_consistent_hashes(evidence):
    builder, folders = evidence
    folder = folders["vllm", "recheck5"]
    prompts = json.loads((folder / "prompts.json").read_text())
    prompts[2]["prompt_tokens"] = [555, 42]
    (folder / "prompts.json").write_text(json.dumps(prompts))
    rows = [json.loads(line) for line in (folder / "samples.jsonl").read_text().splitlines()]
    rows[2]["prompt_token_sha256"] = digest(prompts[2]["prompt_tokens"])
    (folder / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    rewrite_summary(folder, lambda summary: summary.update(prompt_tokens_sha256=digest([p["prompt_tokens"] for p in prompts])))
    report = builder.analyze()
    assert report["status"] == "failed" and report["comparisons"] == []
    assert "first5 prompt token IDs/index" in str(report["errors"])


@pytest.mark.parametrize("fault", ["missing", "partial", "truncated_raw", "wrong_assertion", "row_hash", "row_index"])
def test_missing_partial_and_corrupt_evidence_fail_closed(evidence, fault):
    builder, folders = evidence
    folder = folders["transformers", "recheck5"]
    if fault == "missing":
        (folder / "samples.jsonl").unlink()
    elif fault == "partial":
        rewrite_summary(folder, lambda summary: summary.update(status="running", completed_samples=4))
    elif fault == "wrong_assertion":
        rewrite_summary(folder, lambda summary: summary["environment"].update(caffeinate_assertions="i"))
    else:
        rows = [json.loads(line) for line in (folder / "samples.jsonl").read_text().splitlines()]
        if fault == "truncated_raw":
            rows.pop()
        else:
            rows[0]["prompt_token_sha256" if fault == "row_hash" else "index"] = "bad" if fault == "row_hash" else 9
        (folder / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    report = builder.analyze()
    assert report["status"] == "failed" and report["comparisons"] == []
    assert report["errors"]
