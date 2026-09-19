"""Validate vLLM phase boundaries with an in-process fake stepping engine."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from inference_lab.backends.apple.vllm_backend import VLLMBackend


class Device:
    def __init__(self):
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1

    def reset_peak_memory(self):
        pass

    def get_peak_memory(self):
        return 100_000_000


class Engine:
    def __init__(self, cumulative_outputs, cached_tokens=0):
        self.outputs = cumulative_outputs
        self.cached_tokens = cached_tokens
        self.requests = []

    def add_request(self, request_id, prompt, params):
        self.requests.append((request_id, prompt, params))

    def has_unfinished_requests(self):
        return bool(self.outputs)

    def step(self):
        tokens = self.outputs.pop(0)
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=tokens)],
                                num_cached_tokens=self.cached_tokens)]


def configured_backend(outputs, cached_tokens=0):
    backend = VLLMBackend("/tmp/fake-model")
    backend.engine = Engine(outputs, cached_tokens)
    backend._mx = Device()
    backend._SamplingParams = lambda **kwargs: kwargs
    backend._output_kind = "cumulative"
    backend.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens))
    return backend


def test_chunked_prefill_waits_for_first_token_and_counts_n_minus_one():
    backend = configured_backend([[], [], [11], [11, 12], [11, 12, 13]])
    result = backend.measure([1, 2, 3, 4], 3)
    assert result["generated_token_ids"] == [11, 12, 13]
    assert result["decode_tokens"] == 2
    assert result["prompt_tokens"] == 4
    assert backend._mx.synchronizations == 3
    request_id, prompt, params = backend.engine.requests[0]
    assert prompt == {"prompt_token_ids": [1, 2, 3, 4]}
    assert params["ignore_eos"] is True
    assert params["max_tokens"] == 3
    assert "min_tokens" not in params  # unsupported by vllm-metal; ignore_eos enforces length


def test_multiple_tokens_in_first_output_cannot_be_used_for_phase_split():
    backend = configured_backend([[11, 12], [11, 12, 13]])
    with pytest.raises(RuntimeError, match="phase split would be invalid"):
        backend.measure([1, 2], 3)


def test_unexpected_prefix_cache_hit_invalidates_measurement():
    backend = configured_backend([[11], [11, 12]], cached_tokens=1)
    with pytest.raises(RuntimeError, match="prefix-cache hit"):
        backend.measure([1, 2], 2)


def test_short_output_invalidates_fixed_length_measurement():
    backend = configured_backend([[11], [11, 12]])
    with pytest.raises(RuntimeError, match="Expected 3 output tokens"):
        backend.measure([1, 2], 3)


def test_load_disables_prefix_cache_and_uses_in_process_single_request_engine(monkeypatch):
    constructor = {}
    device = Device()

    def llm(**kwargs):
        constructor.update(kwargs)
        engine = Engine([])
        engine.engine_core = SimpleNamespace(
            engine_core=SimpleNamespace(batch_queue=None, async_scheduling=False)
        )
        return SimpleNamespace(llm_engine=engine)

    vllm = ModuleType("vllm")
    vllm.LLM = llm
    vllm.SamplingParams = lambda **kwargs: kwargs
    sampling = ModuleType("vllm.sampling_params")
    sampling.RequestOutputKind = SimpleNamespace(CUMULATIVE="cumulative")
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    mlx = ModuleType("mlx")
    mlx.core = device
    for name, module in {"vllm": vllm, "vllm.sampling_params": sampling,
                         "transformers": transformers, "mlx": mlx, "mlx.core": device}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    backend = VLLMBackend("/tmp/fake-model", prefill_step_size=123)
    backend.load()
    import os
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert constructor["enable_prefix_caching"] is False
    assert constructor["max_num_seqs"] == 1
    assert constructor["max_num_batched_tokens"] == 123
    assert constructor["enable_chunked_prefill"] is True
    assert constructor["async_scheduling"] is False
    assert constructor["trust_remote_code"] is False


@pytest.mark.parametrize("async_scheduling, batch_queue", [(True, None), (False, [])])
def test_runtime_rejects_async_scheduler_or_forward_batch_queue(async_scheduling, batch_queue):
    backend = configured_backend([])
    backend.engine.engine_core = SimpleNamespace(engine_core=SimpleNamespace(
        async_scheduling=async_scheduling, batch_queue=batch_queue,
    ))
    with pytest.raises(RuntimeError, match="Phase timing requires"):
        backend._validate_synchronous_engine()


def test_runtime_rejects_out_of_process_engine():
    backend = configured_backend([])
    backend.engine.engine_core = SimpleNamespace()
    with pytest.raises(RuntimeError, match="in-process engine core"):
        backend._validate_synchronous_engine()
