"""Protect the shared workload and durable failure reporting without inference."""

import fcntl
import hashlib
import json
from dataclasses import replace

import pytest

from inference_lab.benchmarking import runner as runner_module
from inference_lab.core.config import BenchmarkConfig
from inference_lab.data.prompts import PromptDataset


class RecordingTokenizer:
    def __init__(self):
        self.chat_calls = []
        self.encode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        tokens = [1, *map(ord, messages[0]["content"]), 2]
        # Transformers 5 defaults to a BatchEncoding-like dictionary.
        return tokens if kwargs.get("return_dict") is False else {"input_ids": tokens}

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        return list(map(ord, text))


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "dataset.jsonl"
    rows = [
        {"problem": "What is 2 + 2?", "response": "<think>SECRET_TEACHER_CHAIN"},
        {"problem": "What is 3 + 3?", "response": "<think>OTHER_TEACHER_CHAIN"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def config(dataset):
    return BenchmarkConfig("mlx", dataset_path=str(dataset), count=2,
                           max_new_tokens=4, warmup=0)


def test_problem_mode_never_tokenizes_teacher_response(config):
    tokenizer = RecordingTokenizer()
    samples = PromptDataset(config, tokenizer).load()
    assert tokenizer.encode_calls == []
    assert all("TEACHER" not in messages[0]["content"] for messages, _ in tokenizer.chat_calls)
    for _, kwargs in tokenizer.chat_calls:
        assert kwargs == dict(tokenize=True, add_generation_prompt=True,
                              enable_thinking=True, return_dict=False)
    assert all(sample.teacher_response_chars > 0 for sample in samples)


def test_all_frameworks_get_identical_prompt_token_sequences(config):
    reference = PromptDataset(config, RecordingTokenizer()).load()
    for framework in ("transformers", "vllm", "mlx-vlm"):
        samples = PromptDataset(replace(config, backend=framework), RecordingTokenizer()).load()
        assert [s.prompt_tokens for s in samples] == [s.prompt_tokens for s in reference]
        assert [s.token_sha256 for s in samples] == [s.token_sha256 for s in reference]


def test_chain_prefix_is_explicit_and_token_limited(config):
    tokenizer = RecordingTokenizer()
    problem = PromptDataset(config, RecordingTokenizer()).load()[0]
    chain = PromptDataset(replace(config, prompt_mode="chain-prefix", chain_prefix_tokens=3), tokenizer).load()[0]
    assert chain.prompt_tokens == problem.prompt_tokens + list(map(ord, "SEC"))
    assert tokenizer.encode_calls[0] == ("SECRET_TEACHER_CHAIN", {"add_special_tokens": False})


def test_long_problem_is_rejected_instead_of_truncated(config):
    with pytest.raises(ValueError, match="raise the limit explicitly"):
        PromptDataset(replace(config, max_prompt_tokens=2), RecordingTokenizer()).load()


class FakeBackend:
    def __init__(self, failure=None, bad_decode_count=False, prefill_seconds=1.0):
        self.tokenizer = RecordingTokenizer()
        self.calls = []
        self.failure = failure
        self.bad_decode_count = bad_decode_count
        self.prefill_seconds = prefill_seconds

    def load(self):
        if self.failure == "load":
            raise RuntimeError("cannot load model")

    def metadata(self):
        return {"framework": "fake"}

    def measure(self, tokens, max_new_tokens):
        self.calls.append(list(tokens))
        if self.failure == "second-request" and len(self.calls) == 2:
            raise RuntimeError("device ran out of memory")
        return dict(
            prompt_tokens=len(tokens), generated_tokens=max_new_tokens,
            decode_tokens=max_new_tokens if self.bad_decode_count else max_new_tokens - 1,
            prefill_seconds=self.prefill_seconds, decode_seconds=0.5,
            generated_token_ids=[3] * max_new_tokens, output_text="result",
        )


def prepare_runner(monkeypatch, tmp_path, backend):
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    monkeypatch.setattr(runner_module, "environment", lambda: {"test": True})
    monkeypatch.setattr(runner_module, "create_backend", lambda config: backend)


def read_summary(tmp_path):
    paths = list((tmp_path / "artifacts/runs").glob("*/summary.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text()), paths[0].parent


def test_runner_records_exact_tokens_and_excludes_first_decode_token(monkeypatch, tmp_path, config):
    backend = FakeBackend()
    prepare_runner(monkeypatch, tmp_path, backend)
    runner_module.BenchmarkRunner(config).run()
    summary, directory = read_summary(tmp_path)
    assert summary["status"] == "completed"
    assert summary["metrics"]["decode"]["total_tokens"] == 6
    assert summary["metrics"]["generated_tokens"] == 8
    prompts = json.loads((directory / "prompts.json").read_text())
    assert backend.calls == [prompt["prompt_tokens"] for prompt in prompts]
    assert summary["prompt_tokens_sha256"] == hashlib.sha256(json.dumps(backend.calls).encode()).hexdigest()


def test_experimental_backend_factory_preserves_shared_measurement_contract(monkeypatch, tmp_path, config):
    backend = FakeBackend()
    prepare_runner(monkeypatch, tmp_path, FakeBackend(failure="load"))
    runner_module.BenchmarkRunner(config, backend_factory=lambda _: backend).run()
    summary, _ = read_summary(tmp_path)
    assert summary["status"] == "completed"
    assert summary["metrics"]["decode"]["total_tokens"] == 6
    assert len(backend.calls) == config.count


@pytest.mark.parametrize("failure, expected_rows", [("load", 0), ("second-request", 1)])
def test_backend_failure_persists_status_and_finished_rows(monkeypatch, tmp_path, config, failure, expected_rows):
    prepare_runner(monkeypatch, tmp_path, FakeBackend(failure=failure))
    with pytest.raises(RuntimeError, match="Benchmark failed"):
        runner_module.BenchmarkRunner(config).run()
    summary, directory = read_summary(tmp_path)
    assert summary["status"] == "failed"
    assert summary["completed_samples"] == expected_rows
    assert summary["error_type"] == "RuntimeError"
    assert (directory / "error.txt").is_file()
    if expected_rows:
        assert len((directory / "samples.jsonl").read_text().splitlines()) == expected_rows


def test_incorrect_decode_accounting_is_rejected(monkeypatch, tmp_path, config):
    prepare_runner(monkeypatch, tmp_path, FakeBackend(bad_decode_count=True))
    with pytest.raises(RuntimeError, match="Benchmark failed"):
        runner_module.BenchmarkRunner(config).run()
    summary, _ = read_summary(tmp_path)
    assert summary["status"] == "failed"
    assert summary["completed_samples"] == 0
    assert "exclude the first token" in summary["error"]


@pytest.mark.parametrize("malformation", ["short-output", "missing-ids", "wrong-id-count", "invalid-id"])
def test_runner_rejects_noncomparable_output_lengths(monkeypatch, tmp_path, config, malformation):
    backend = FakeBackend()
    original_measure = backend.measure

    def malformed_measure(tokens, max_new_tokens):
        row = original_measure(tokens, max_new_tokens)
        if malformation == "short-output":
            row.update(generated_tokens=3, decode_tokens=2, generated_token_ids=[3, 3, 3])
        elif malformation == "missing-ids":
            row.pop("generated_token_ids")
        elif malformation == "wrong-id-count":
            row["generated_token_ids"] = [3, 3, 3]
        else:
            row["generated_token_ids"] = [3, 3, 3, True]
        return row

    backend.measure = malformed_measure
    prepare_runner(monkeypatch, tmp_path, backend)
    with pytest.raises(RuntimeError, match="Benchmark failed"):
        runner_module.BenchmarkRunner(config).run()
    summary, _ = read_summary(tmp_path)
    assert summary["status"] == "failed"
    assert summary["completed_samples"] == 0
    assert summary["error_type"] == "ValueError"


@pytest.mark.parametrize("duration", [0.0, float("nan"), float("inf")])
def test_invalid_durations_cannot_leave_running_or_completed_summary(monkeypatch, tmp_path, config, duration):
    prepare_runner(monkeypatch, tmp_path, FakeBackend(prefill_seconds=duration))
    with pytest.raises(RuntimeError, match="Benchmark failed"):
        runner_module.BenchmarkRunner(config).run()
    summary, _ = read_summary(tmp_path)
    assert summary["status"] == "failed"
    assert summary["completed_samples"] == 0


def test_lock_contention_is_persisted_as_failed_run(monkeypatch, tmp_path, config):
    prepare_runner(monkeypatch, tmp_path, FakeBackend())
    lock_path = tmp_path / "artifacts/gpu.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError):
            runner_module.BenchmarkRunner(config).run()
    summary, _ = read_summary(tmp_path)
    assert summary["status"] == "failed"
    assert "GPU lock" in summary["error"]


@pytest.mark.parametrize("failure", [None, "load", "second-request"])
def test_user_activity_spans_work_and_is_ended_before_final_summary(monkeypatch, tmp_path, config, failure):
    from inference_lab.core.activity import UserInitiatedActivity
    events = []

    class API:
        def begin(self, options, reason):
            assert options == 0x00FFFFFF
            events.append("begin")
            return object()

        def end(self, handle):
            events.append("end")

    backend = FakeBackend(failure=failure)
    prepare_runner(monkeypatch, tmp_path, backend)
    monkeypatch.setattr(runner_module, "UserInitiatedActivity", lambda enabled, reason: (
        UserInitiatedActivity(enabled, reason, _api_factory=API)
    ))

    def resources():
        assert events == ["begin"]
        return {}

    monkeypatch.setattr(runner_module, "resource_snapshot", resources)
    if failure:
        with pytest.raises(RuntimeError, match="Benchmark failed"):
            runner_module.BenchmarkRunner(replace(config, user_initiated=True)).run()
    else:
        runner_module.BenchmarkRunner(replace(config, user_initiated=True)).run()
    assert events == ["begin", "end"]
    summary, _ = read_summary(tmp_path)
    assert summary["config"]["user_initiated"] is True
    assert summary["activity"]["user_initiated"] is True
    assert summary["activity"]["options"] == 0x00FFFFFF
    assert summary["activity"]["started"] is True
    assert summary["activity"]["ended"] is True
    assert summary["activity"]["active"] is False
