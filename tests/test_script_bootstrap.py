"""Exercise each Python entry point without importing any inference library."""

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script_name, expected_backend", [
    ("mlx", "mlx"), ("mlx_vlm", "mlx-vlm"),
    ("transformers", "transformers"), ("vllm", "vllm"),
    ("speculative", "speculative"), ("mtp", "mtp"),
])
def test_library_named_scripts_do_not_shadow_installed_packages(tmp_path, script_name, expected_backend):
    # Four lightweight stand-ins make the test independent of optional packages.
    for name in ("mlx", "mlx_vlm", "transformers", "vllm"):
        (tmp_path / f"{name}.py").write_text("raise AssertionError('must only find, not import')\n")
    script = ROOT / "scripts/benchmark" / f"{script_name}.py"
    code = """
import importlib.util
from pathlib import Path
import runpy
import sys
from types import ModuleType

script, libraries, source, expected = map(str, sys.argv[1:])
sys.path[:0] = [str(Path(script).parent), libraries]

def check(backend=None):
    assert backend == (None if expected in ('speculative', 'mtp') else expected)
    assert Path(sys.path[0]) == Path(source)
    assert Path(script).parent not in [Path(p or '.').resolve() for p in sys.path]
    for name in ('mlx', 'mlx_vlm', 'transformers', 'vllm'):
        spec = importlib.util.find_spec(name)
        assert Path(spec.origin) == Path(libraries) / (name + '.py'), (name, spec.origin)

module_name = ('inference_lab.benchmarking.' + expected if expected in ('speculative', 'mtp')
               else 'inference_lab.benchmarking.cli')
cli = ModuleType(module_name)
cli.main = check
sys.modules[module_name] = cli
runpy.run_path(script, run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(script), str(tmp_path), str(ROOT / "src"), expected_backend],
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
