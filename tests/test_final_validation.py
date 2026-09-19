"""Audit synthetic archived runs; no model, network or GPU imports."""

import hashlib
import json
from copy import deepcopy

import pytest

from inference_lab.benchmarking.metrics import DISPERSION_FIELDS, aggregate
from inference_lab.benchmarking.validation import FinalSeriesValidator


def token_hash(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def make_run(runs, backend, stamp="20260918T220000.000000Z", tokens=128):
    directory = runs / f"{stamp}-{backend}-awake"
    directory.mkdir(parents=True)
    prompts = [{"index": index, "prompt_tokens": [1, 2, index], "original_prompt_tokens": 3}
               for index in range(100)]
    rows = [{"index": index, "prompt_tokens": 3, "generated_tokens": tokens,
             "decode_tokens": tokens - 1, "generated_token_ids": list(range(tokens)),
             "prefill_seconds": 0.5 + index / 100, "decode_seconds": 2.0 + index / 100,
             "peak_memory_gb": 5.0, "prompt_token_sha256": token_hash(prompt["prompt_tokens"])}
            for index, prompt in enumerate(prompts)]
    summary = {"status": "completed", "started_at": stamp, "completed_samples": 100,
               "config": {"backend": backend, "label": "awake", "count": 100,
                          "max_new_tokens": tokens, "prompt_mode": "problem"},
               "environment": {"caffeinate_assertions": "di"},
               "dataset_sha256": "a" * 64, "model_manifest_sha256": "b" * 64,
               "prompt_tokens_sha256": token_hash([prompt["prompt_tokens"] for prompt in prompts]),
               "metrics": aggregate(rows), "run_directory": str(directory)}
    save_run(directory, summary, prompts, rows)
    return directory, summary, prompts, rows


def save_run(directory, summary, prompts, rows):
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "prompts.json").write_text(json.dumps(prompts))
    (directory / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def validate(tmp_path):
    return FinalSeriesValidator(tmp_path / "runs", output=tmp_path / "validation.json").run()


@pytest.mark.parametrize("tokens", [128, 256])
def test_valid_series_recalculates_metrics_and_accepts_common_future_length(tmp_path, tokens):
    for backend in FinalSeriesValidator.BACKENDS:
        make_run(tmp_path / "runs", backend, tokens=tokens)
    result = validate(tmp_path)
    assert result["status"] == "passed"
    assert result["errors"] == []
    assert result["common"]["max_new_tokens"] == tokens
    assert len(result["runs"]) == 4
    assert all(run["generated_token_ids_checked"] == 100 * tokens for run in result["runs"])
    assert all(run["recalculated_metrics"]["decode"]["total_tokens"] == 100 * (tokens - 1)
               for run in result["runs"])
    assert json.loads((tmp_path / "validation.json").read_text()) == result


@pytest.mark.parametrize("mutation, message", [
    ("missing_row", "exactly 100"),
    ("duplicate_index", "indices"),
    ("output_length", "output token IDs"),
    ("decode_count", "decode_tokens"),
    ("prompt_count", "prompt-token count"),
    ("prompt_hash", "prompt-token SHA256"),
    ("negative_duration", "positive and finite"),
    ("nan_duration", "positive and finite"),
    ("metric_mismatch", "recalculated"),
    ("global_hash_mismatch", "matching valid dataset_sha256"),
    ("length_mismatch", "matching valid max_new_tokens"),
])
def test_rejects_corrupt_or_incomparable_evidence(tmp_path, mutation, message):
    for backend in FinalSeriesValidator.BACKENDS:
        record = make_run(tmp_path / "runs", backend)
        if backend == "vllm":
            directory, summary, prompts, rows = record
    if mutation == "missing_row": rows.pop()
    elif mutation == "duplicate_index": rows[1]["index"] = 0
    elif mutation == "output_length": rows[0]["generated_token_ids"].pop()
    elif mutation == "decode_count": rows[0]["decode_tokens"] = 128
    elif mutation == "prompt_count": rows[0]["prompt_tokens"] = 2
    elif mutation == "prompt_hash": rows[0]["prompt_token_sha256"] = "c" * 64
    elif mutation == "negative_duration": rows[0]["prefill_seconds"] = -1.0
    elif mutation == "nan_duration": rows[0]["decode_seconds"] = float("nan")
    elif mutation == "metric_mismatch": summary["metrics"]["decode"]["mean_tokens_per_second"] += 1
    elif mutation == "global_hash_mismatch": summary["dataset_sha256"] = "c" * 64
    elif mutation == "length_mismatch": summary["config"]["max_new_tokens"] = 256
    save_run(directory, summary, prompts, rows)
    result = validate(tmp_path)
    assert result["status"] == "failed"
    assert any(message in error for error in result["errors"])


def test_incomplete_series_never_claims_passed(tmp_path):
    for backend in FinalSeriesValidator.BACKENDS[:-1]:
        make_run(tmp_path / "runs", backend)
    directory, summary, prompts, rows = make_run(tmp_path / "runs", "vllm")
    summary.update(status="running", completed_samples=20)
    save_run(directory, summary, prompts, rows[:20])
    result = validate(tmp_path)
    assert result["status"] == "failed"
    assert any("Incomplete series" in error and "vllm" in error for error in result["errors"])


def test_latest_completed_run_is_audited_without_falling_back_on_older_valid_run(tmp_path):
    for backend in FinalSeriesValidator.BACKENDS:
        make_run(tmp_path / "runs", backend)
    directory, summary, prompts, rows = make_run(tmp_path / "runs", "vllm", "20260919T220000.000000Z")
    rows[0]["generated_token_ids"] = []
    save_run(directory, summary, prompts, rows)
    result = validate(tmp_path)
    assert result["status"] == "failed"
    selected = next(run for run in result["runs"] if run["backend"] == "vllm")
    assert selected["run_directory"] == str(directory)


def test_missing_caffeinate_assertion_does_not_qualify(tmp_path):
    for backend in FinalSeriesValidator.BACKENDS:
        directory, summary, prompts, rows = make_run(tmp_path / "runs", backend)
    summary["environment"]["caffeinate_assertions"] = "i"
    save_run(directory, summary, prompts, rows)
    result = validate(tmp_path)
    assert result["status"] == "failed"
    assert "vllm" not in result["selected_backends"]


def test_legacy_summaries_remain_valid_and_unmodified_while_audit_adds_dispersion(tmp_path):
    originals = {}
    for backend in FinalSeriesValidator.BACKENDS:
        directory, summary, prompts, rows = make_run(tmp_path / "runs", backend)
        for phase in ("prefill", "decode"):
            for field in DISPERSION_FIELDS:
                del summary["metrics"][phase][field]
        save_run(directory, summary, prompts, rows)
        originals[directory / "summary.json"] = (directory / "summary.json").read_bytes()
    result = validate(tmp_path)
    assert result["status"] == "passed"
    for run in result["runs"]:
        for phase in ("prefill", "decode"):
            assert DISPERSION_FIELDS <= run["recalculated_metrics"][phase].keys()
    assert all(path.read_bytes() == before for path, before in originals.items())


@pytest.mark.parametrize("field", sorted(DISPERSION_FIELDS))
def test_stored_dispersion_fields_are_checked_if_present(tmp_path, field):
    rows = [dict(prompt_tokens=100, prefill_seconds=1, generated_tokens=11, decode_tokens=10, decode_seconds=1),
            dict(prompt_tokens=100, prefill_seconds=10, generated_tokens=11, decode_tokens=10, decode_seconds=2)]
    expected = aggregate(rows)
    stored = deepcopy(expected)
    stored["prefill"][field] += 1
    validator = FinalSeriesValidator(output=tmp_path / "validation.json")
    with pytest.raises(ValueError, match=field):
        validator._compare_metrics(stored, expected)


def test_legacy_compatibility_does_not_make_original_metrics_optional(tmp_path):
    rows = [dict(prompt_tokens=100, prefill_seconds=1, generated_tokens=11, decode_tokens=10, decode_seconds=1)]
    expected = aggregate(rows)
    stored = deepcopy(expected)
    del stored["prefill"]["total_seconds"]
    validator = FinalSeriesValidator(output=tmp_path / "validation.json")
    with pytest.raises(ValueError, match="Missing metrics.prefill.total_seconds"):
        validator._compare_metrics(stored, expected)


@pytest.mark.parametrize("field", ["sample_variance_tokens_per_second", "sample_stddev_tokens_per_second",
                                   "coefficient_of_variation_percent"])
def test_single_request_nullable_statistics_are_validated_strictly(tmp_path, field):
    rows = [dict(prompt_tokens=100, prefill_seconds=1, generated_tokens=11, decode_tokens=10, decode_seconds=1)]
    expected = aggregate(rows)
    validator = FinalSeriesValidator(output=tmp_path / "validation.json")
    validator._compare_metrics(deepcopy(expected), expected)
    stored = deepcopy(expected)
    stored["decode"][field] = 0.0
    with pytest.raises(ValueError, match=field):
        validator._compare_metrics(stored, expected)


def test_multi_request_numeric_statistic_cannot_be_replaced_by_null(tmp_path):
    rows = [dict(prompt_tokens=100, prefill_seconds=1, generated_tokens=11, decode_tokens=10, decode_seconds=1),
            dict(prompt_tokens=100, prefill_seconds=2, generated_tokens=11, decode_tokens=10, decode_seconds=2)]
    expected = aggregate(rows)
    stored = deepcopy(expected)
    stored["prefill"]["sample_variance_tokens_per_second"] = None
    validator = FinalSeriesValidator(output=tmp_path / "validation.json")
    with pytest.raises(ValueError, match="sample_variance_tokens_per_second"):
        validator._compare_metrics(stored, expected)
