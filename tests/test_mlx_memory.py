"""Mock-only checks for optional, per-request MLX wired-memory limits."""

import json
from types import SimpleNamespace

import pytest

from inference_lab.backends.apple.mlx_backend import MLXBackend
from inference_lab.backends.apple.mlx_memory import RecommendedWiredMemory
from inference_lab.backends.apple.mlx_vlm_backend import MLXVLMBackend
from inference_lab.backends.base import create_backend
from inference_lab.core.config import BenchmarkConfig


class MemoryRuntime:
    def __init__(self, initial_limit=0):
        self.limit = initial_limit
        self.events = []
        self.synchronize_count = 0
        self.fail_synchronize_at = None

    def device_info(self):
        return {"max_recommended_working_set_size": 4096}

    def synchronize(self):
        self.events.append("synchronize")
        self.synchronize_count += 1
        if self.synchronize_count == self.fail_synchronize_at:
            raise RuntimeError("mock synchronization error")

    def set_wired_limit(self, limit):
        previous, self.limit = self.limit, limit
        self.events.append(("set_wired_limit", limit))
        return previous


@pytest.mark.parametrize("backend_type", [MLXBackend, MLXVLMBackend])
def test_optional_scope_does_not_change_measurement_values(backend_type):
    runtime = MemoryRuntime(initial_limit=1024)
    backend = backend_type("/tmp/fake-model", wired_memory=True)
    backend._model = object()
    backend._mx = runtime
    result = {"prefill_seconds": 1.5, "decode_seconds": 3.5}

    def measured(tokens, count):
        assert runtime.limit == 4096
        runtime.events.append("measure")
        return result

    backend._measure = measured
    assert backend.measure([7], 2) is result
    assert runtime.events == ["synchronize", ("set_wired_limit", 4096), "measure",
                              "synchronize", ("set_wired_limit", 1024)]
    assert runtime.limit == 1024
    assert backend.metadata()["wired_memory"] is True


def test_disabled_policy_never_changes_or_synchronizes_runtime():
    backend = MLXBackend("/tmp/fake-model")
    runtime = MemoryRuntime()
    backend._model, backend._mx = object(), runtime
    backend._measure = lambda tokens, count: {"unchanged": True}
    assert backend.measure([7], 2) == {"unchanged": True}
    assert runtime.events == []
    assert backend.metadata()["wired_memory"] is False


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_request_failure_restores_previous_zero_limit(error):
    runtime = MemoryRuntime(initial_limit=0)
    with pytest.raises(error):
        with RecommendedWiredMemory(runtime):
            assert runtime.limit == 4096
            raise error("request failed")
    assert runtime.limit == 0
    assert runtime.events[-2:] == ["synchronize", ("set_wired_limit", 0)]


def test_cleanup_synchronization_failure_still_restores_limit():
    runtime = MemoryRuntime(initial_limit=512)
    runtime.fail_synchronize_at = 2
    with pytest.raises(RuntimeError, match="synchronization"):
        with RecommendedWiredMemory(runtime):
            pass
    assert runtime.limit == 512
    assert runtime.events[-1] == ("set_wired_limit", 512)


def test_enter_synchronization_failure_does_not_change_limit():
    runtime = MemoryRuntime(initial_limit=512)
    runtime.fail_synchronize_at = 1
    with pytest.raises(RuntimeError, match="synchronization"):
        with RecommendedWiredMemory(runtime):
            pytest.fail("request must not start")
    assert runtime.limit == 512
    assert runtime.events == ["synchronize"]


@pytest.mark.parametrize("backend", ["transformers", "vllm"])
def test_unsupported_frameworks_explicitly_reject_wired_memory(backend):
    with pytest.raises(ValueError, match="only by MLX.*MLX-VLM"):
        BenchmarkConfig(backend, wired_memory=True)
    config = SimpleNamespace(backend=backend, wired_memory=True, model_path="unused",
                             prefill_step_size=512, kv_bits=None)
    with pytest.raises(ValueError, match="only by MLX and MLX-VLM"):
        create_backend(config)


@pytest.mark.parametrize("backend", ["mlx", "mlx-vlm"])
def test_factory_passes_policy_to_supported_backend(backend):
    adapter = create_backend(BenchmarkConfig(backend, wired_memory=True))
    assert adapter.wired_memory is True


@pytest.mark.parametrize("configured, arguments, expected", [
    (False, [], False), (False, ["--wired-memory"], True),
    (True, ["--no-wired-memory"], False),
])
def test_cli_keeps_baseline_default_and_supports_explicit_ab_policy(monkeypatch, tmp_path, configured, arguments, expected):
    from inference_lab.benchmarking import cli
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"wired_memory": configured}))
    seen = []
    monkeypatch.setattr(cli, "BenchmarkRunner", lambda settings: SimpleNamespace(run=lambda: seen.append(settings)))
    monkeypatch.setattr("sys.argv", ["mlx.py", "--config", str(config), *arguments])
    cli.main("mlx")
    assert seen[0].wired_memory is expected
