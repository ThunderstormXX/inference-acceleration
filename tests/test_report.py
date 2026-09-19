"""Keep smoke-test rates and interrupted partial results out of the 100-row table."""

from dataclasses import asdict
import csv
import json
from statistics import mean, stdev, variance

import pytest

from inference_lab.benchmarking import report as report_module
from inference_lab.core.config import BenchmarkConfig


def test_full_comparison_separates_smoke_and_hides_partial_failure_rates(monkeypatch, tmp_path):
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    runs = tmp_path / "artifacts/runs"
    metrics = {
        "prefill": {"mean_tokens_per_second": 100.0, "aggregate_tokens_per_second": 99.0},
        "decode": {"mean_tokens_per_second": 30.0, "aggregate_tokens_per_second": 29.0},
    }
    cases = [
        ("full", "completed", 100, 100),
        ("smoke", "completed", 2, 2),
        ("interrupted", "failed", 17, 100),
    ]
    for label, status, completed, requested in cases:
        directory = runs / label
        directory.mkdir(parents=True)
        summary = dict(
            config=asdict(BenchmarkConfig("mlx", label=label, count=requested)),
            status=status, completed_samples=completed, metrics=metrics,
            prompt_tokens_sha256=f"hash-{label}", dataset_sha256="same-dataset",
            model_manifest_sha256="same-weights",
            run_directory=str(directory),
        )
        (directory / "summary.json").write_text(json.dumps(summary))
    records = report_module.BenchmarkReport(runs).build()
    text = (tmp_path / "artifacts/reports/comparison.md").read_text()
    primary, diagnostics = text.split("## Проверочные и незавершённые запуски", 1)
    assert "mlx / full" in primary
    assert "mlx / smoke" not in primary
    assert "mlx / interrupted" not in primary
    assert "mlx / smoke" in diagnostics
    assert "mlx / interrupted | failed | 17/100 | 128 | — | —" in diagnostics
    failed = next(row for row in records if row["label"] == "interrupted")
    assert failed["prefill_mean_tok_s"] is None
    assert failed["decode_mean_tok_s"] is None
    assert failed["prefill_aggregate_tok_s"] is None
    assert failed["decode_aggregate_tok_s"] is None
    assert all(row["prompt_tokens_sha256"] for row in records)
    assert all(row["model_manifest_sha256"] == "same-weights" for row in records)
    assert "## Полные прогоны на 100 примерах" in primary
    assert "Итоговая серия awake" not in text


def test_completed_awake_series_precedes_history_and_exports_all_conditions(monkeypatch, tmp_path):
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    runs = tmp_path / "artifacts/runs"
    metrics = {
        "prefill": {"mean_tokens_per_second": 110.0, "aggregate_tokens_per_second": 109.0},
        "decode": {"mean_tokens_per_second": 31.0, "aggregate_tokens_per_second": 30.0},
    }
    cases = [
        ("01-baseline", "mlx", "baseline", "completed", 100, 100, None),
        ("02-awake", "mlx", "awake", "completed", 100, 100, "di"),
        ("03-no-display-assertion", "vllm", "awake", "completed", 100, 100, "i"),
        ("04-failed", "transformers", "awake", "failed", 100, 100, "di"),
        ("05-partial", "transformers", "awake", "completed", 17, 100, "di"),
        ("06-wrong-request-count", "vllm", "awake", "completed", 100, 200, "di"),
        ("07-smoke", "mlx-vlm", "awake", "completed", 1, 1, "di"),
    ]
    for name, backend, label, status, completed, requested, assertions in cases:
        directory = runs / name
        directory.mkdir(parents=True)
        config = asdict(BenchmarkConfig(backend, label=label, count=min(requested, 100)))
        # The report must also handle an inconsistent externally written summary.
        config["count"] = requested
        config.update(wired_memory=backend == "mlx", user_initiated=True)
        summary = dict(
            config=config, status=status, completed_samples=completed, metrics=metrics,
            environment={"caffeinate_assertions": assertions},
            prompt_tokens_sha256="same-prompts", dataset_sha256="same-dataset",
            model_manifest_sha256="same-weights", run_directory=str(directory),
        )
        (directory / "summary.json").write_text(json.dumps(summary))
    records = report_module.BenchmarkReport(runs).build()
    output = tmp_path / "artifacts/reports"
    text = (output / "comparison.md").read_text()
    first, remainder = text.split("## История полных прогонов на 100 примерах", 1)
    history, diagnostics = remainder.split("## Проверочные и незавершённые запуски", 1)
    assert "## Итоговая серия awake: 100 примеров, caffeinate -di" in first
    assert "mlx / awake | completed | 100/100 |" in first
    assert "baseline" not in first
    assert "vllm / awake" not in first
    assert "transformers / awake" not in first
    assert "mlx-vlm / awake" not in first
    assert "mlx / baseline | completed | 100/100 |" in history
    assert "vllm / awake | completed | 100/100 |" in history
    assert "transformers / awake | failed | 100/100 | 128 | — | —" in diagnostics
    assert "transformers / awake | completed | 17/100 | 128 | — | —" in diagnostics
    assert "vllm / awake | completed | 100/200 | 128 | — | —" in diagnostics
    assert "mlx-vlm / awake | completed | 1/1 |" in diagnostics
    exported = json.loads((output / "comparison.json").read_text())
    with (output / "comparison.csv").open(newline="") as stream:
        csv_records = list(csv.DictReader(stream))
    assert len(records) == len(exported) == len(csv_records) == len(cases)
    awake = next(row for row in exported if row["directory"].endswith("02-awake"))
    assert awake["caffeinate_assertions"] == "di"
    assert awake["wired_memory"] is True
    assert awake["user_initiated"] is True
    assert next(row for row in csv_records if row["directory"].endswith("02-awake"))["caffeinate_assertions"] == "di"


def test_incomplete_awake_does_not_replace_existing_full_table(monkeypatch, tmp_path):
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    runs = tmp_path / "artifacts/runs"
    directory = runs / "awake-still-running"
    directory.mkdir(parents=True)
    summary = dict(
        config=asdict(BenchmarkConfig("transformers", label="awake", count=100)),
        status="running", completed_samples=16, metrics={},
        environment={"caffeinate_assertions": "di"}, run_directory=str(directory),
    )
    (directory / "summary.json").write_text(json.dumps(summary))
    report_module.BenchmarkReport(runs).build()
    text = (tmp_path / "artifacts/reports/comparison.md").read_text()
    assert "## Полные прогоны на 100 примерах" in text
    assert "Итоговая серия awake" not in text
    assert "История полных прогонов" not in text
    assert "transformers / awake | running | 16/100 | 128 | — | —" in text


def write_archived_raw_run(tmp_path, rows):
    """Write the old on-disk schema: raw request rows but no saved dispersion."""
    directory = tmp_path / "artifacts/runs/archived"
    directory.mkdir(parents=True)
    metrics = {"samples": len(rows)}
    for phase, tokens in (("prefill", "prompt_tokens"), ("decode", "decode_tokens")):
        durations = [row[f"{phase}_seconds"] for row in rows]
        metrics[phase] = {
            "mean_tokens_per_second": mean(row[tokens] / row[f"{phase}_seconds"] for row in rows),
            "aggregate_tokens_per_second": sum(row[tokens] for row in rows) / sum(durations),
        }
    summary = {
        "config": asdict(BenchmarkConfig("mlx", count=len(rows), max_new_tokens=11)),
        "status": "completed", "completed_samples": len(rows), "metrics": metrics,
        "run_directory": str(directory),
    }
    summary_path = directory / "summary.json"
    raw_path = directory / "samples.jsonl"
    summary_path.write_text(json.dumps(summary))
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return summary_path, raw_path, summary


def raw_rows():
    return [
        dict(prompt_tokens=100, generated_tokens=11, decode_tokens=10,
             prefill_seconds=prefill, decode_seconds=decode)
        for prefill, decode in [(1.0, 1.0), (2.0, 2.0), (4.0, 5.0)]
    ]


def test_archived_raw_dispersion_exports_without_changing_means_or_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    rows = raw_rows()
    summary_path, raw_path, summary = write_archived_raw_run(tmp_path, rows)
    # A negligible archived floating-point difference remains byte-for-byte
    # intact in both the source summary and the exported historical mean.
    summary["metrics"]["prefill"]["mean_tokens_per_second"] += 1e-11
    summary_path.write_text(json.dumps(summary))
    original_summary, original_raw = summary_path.read_bytes(), raw_path.read_bytes()
    output = tmp_path / "artifacts/reports"
    output.mkdir(parents=True)
    (output / "recheck-5.md").write_text("# Independent five-prompt recheck\n")

    records = report_module.BenchmarkReport(summary_path.parent.parent).build()

    assert summary_path.read_bytes() == original_summary
    assert raw_path.read_bytes() == original_raw
    exported = json.loads((output / "comparison.json").read_text())[0]
    with (output / "comparison.csv").open(newline="") as stream:
        csv_row = next(csv.DictReader(stream))
    text = (output / "comparison.md").read_text()
    assert "[Повторный замер четырёх framework на пяти примерах](recheck-5.md)" in text
    assert "N−1" in text
    assert records[0] == exported
    for phase, tokens in (("prefill", "prompt_tokens"), ("decode", "decode_tokens")):
        rates = [row[tokens] / row[f"{phase}_seconds"] for row in rows]
        expected = {
            "sample_variance_tokens_per_second": variance(rates),
            "sample_stddev_tokens_per_second": stdev(rates),
            "min_tokens_per_second": min(rates),
            "max_tokens_per_second": max(rates),
            "coefficient_of_variation_percent": 100.0 * stdev(rates) / mean(rates),
        }
        assert exported[f"{phase}_mean_tok_s"] == summary["metrics"][phase]["mean_tokens_per_second"]
        assert exported[f"{phase}_aggregate_tok_s"] == summary["metrics"][phase]["aggregate_tokens_per_second"]
        for suffix, expected_value in expected.items():
            key = f"{phase}_{suffix}"
            assert exported[key] == pytest.approx(expected_value)
            assert float(csv_row[key]) == pytest.approx(expected_value)
        assert f"{mean(rates):.2f} ± {stdev(rates):.2f}" in text


@pytest.mark.parametrize("field", ["mean_tokens_per_second", "aggregate_tokens_per_second"])
def test_archived_rate_disagreement_fails_loudly(monkeypatch, tmp_path, field):
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    summary_path, _, summary = write_archived_raw_run(tmp_path, raw_rows())
    summary["metrics"]["decode"][field] += 1.0
    summary_path.write_text(json.dumps(summary))
    before = summary_path.read_bytes()
    with pytest.raises(ValueError, match=f"archived decode.{field}"):
        report_module.BenchmarkReport(summary_path.parent.parent).build()
    assert summary_path.read_bytes() == before


def test_missing_raw_request_is_not_reported_as_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    summary_path, raw_path, _ = write_archived_raw_run(tmp_path, raw_rows())
    raw_path.write_text("\n".join(raw_path.read_text().splitlines()[:2]) + "\n")
    with pytest.raises(ValueError, match="sample count"):
        report_module.BenchmarkReport(summary_path.parent.parent).build()


def test_single_raw_request_has_undefined_sample_dispersion(monkeypatch, tmp_path):
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    summary_path, _, _ = write_archived_raw_run(tmp_path, raw_rows()[:1])
    row = report_module.BenchmarkReport(summary_path.parent.parent).build()[0]
    for phase in ("prefill", "decode"):
        for field in ("sample_variance_tokens_per_second", "sample_stddev_tokens_per_second",
                      "coefficient_of_variation_percent"):
            assert row[f"{phase}_{field}"] is None
        assert row[f"{phase}_min_tokens_per_second"] == row[f"{phase}_max_tokens_per_second"]
    text = (tmp_path / "artifacts/reports/comparison.md").read_text()
    assert "100.00 ± —" in text
    assert "10.00 ± —" in text
    assert "recheck-5.md" not in text


def test_no_raw_file_preserves_saved_metrics_with_unknown_dispersion(monkeypatch, tmp_path):
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    summary_path, raw_path, summary = write_archived_raw_run(tmp_path, raw_rows())
    raw_path.unlink()
    row = report_module.BenchmarkReport(summary_path.parent.parent).build()[0]
    for phase in ("prefill", "decode"):
        assert row[f"{phase}_mean_tok_s"] == summary["metrics"][phase]["mean_tokens_per_second"]
        assert all(row[f"{phase}_{field}"] is None for field in report_module.DISPERSION_EXPORT_FIELDS)
