"""The paired native MTP entry point must compare the same target runtime."""
import sys
from pathlib import Path
import subprocess

import pytest

from inference_lab.benchmarking import mtp


@pytest.mark.parametrize("override", [
    {"block_size": 1}, {"block_size": 6}, {"block_size": True},
    {"kv_bits": 4}, {"wired_memory": False}, {"backend": "mlx"},
])
def test_unsupported_mtp_config_rejected(override):
    options = dict(backend="mlx-mtp", wired_memory=True)
    options.update(override)
    with pytest.raises(ValueError):
        mtp.MTPConfig(**options)


@pytest.mark.parametrize("mode,backend,label", [
    ("baseline", "mlx-vlm", "mtp-baseline"), ("mtp", "mlx-mtp", "mtp-k3"),
])
def test_cli_pair_defaults_and_backend_factory(monkeypatch, mode, backend, label):
    captured = []

    class Runner:
        def __init__(self, config, backend_factory=None):
            captured.append((config, backend_factory))

        def run(self):
            return None

    monkeypatch.setattr(mtp, "BenchmarkRunner", Runner)
    monkeypatch.setattr(sys, "argv", ["mtp.py", "--mode", mode])
    assert mtp.main() == 0
    config, factory = captured[0]
    assert config.backend == backend
    assert config.label == label
    assert config.count == 5
    assert config.max_new_tokens == 128
    assert config.wired_memory is True
    assert config.prefill_step_size == 512
    if mode == "mtp":
        assert config.block_size == 3
    assert factory is (None if mode == "baseline" else mtp.create_mtp_backend)


def test_python_entrypoint_help_avoids_local_library_shadowing():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/benchmark/mtp.py"), "--help"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert "--mode {baseline,mtp}" in result.stdout
    assert "--block-size" in result.stdout
