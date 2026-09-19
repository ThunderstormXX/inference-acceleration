import pytest

from inference_lab.benchmarking.speculative import SpeculativeConfig


@pytest.mark.parametrize("override", [
    {"block_size": 1}, {"block_size": 16}, {"block_size": 3.0},
    {"draft_bits": 3}, {"draft_bits": 4.0}, {"kv_bits": 4},
    {"wired_memory": False}, {"backend": "mlx"},
])
def test_unsupported_speculative_protocol_is_rejected(override):
    options = dict(backend="mlx-dflash", wired_memory=True)
    options.update(override)
    with pytest.raises(ValueError):
        SpeculativeConfig(**options)


def test_speculative_protocol_retains_fixed_length_phase_convention():
    config = SpeculativeConfig(backend="mlx-dflash", wired_memory=True, count=5)
    assert config.max_new_tokens == 128
    assert config.block_size == 3
    assert config.draft_bits == 4
    assert config.prompt_mode == "problem"
