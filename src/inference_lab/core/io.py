"""Small deterministic I/O helpers."""
import hashlib
import json
import os
import platform
import resource
import subprocess
from importlib.metadata import distributions
from pathlib import Path


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def power_thermal_snapshot() -> dict:
    """Read only power source and thermal category, outside inference timing."""
    result = {"power_source": {"status": "unavailable"},
              "thermal_state": {"status": "unavailable"}}
    if platform.system() != "Darwin":
        return result
    try:
        power = subprocess.check_output(["pmset", "-g", "batt"], text=True,
                                        stderr=subprocess.PIPE, timeout=3).strip()
        result["power_source"] = {"status": "available", "source": "pmset -g batt",
                                  "raw": power}
    except (OSError, subprocess.SubprocessError) as exc:
        result["power_source"]["error"] = str(exc)
    try:
        from inference_lab.diagnostics.host import HostDiagnostics
        result["thermal_state"] = {"status": "available", **HostDiagnostics.thermal_state()}
    except Exception as exc:
        result["thermal_state"]["error"] = str(exc)
    return result


def environment() -> dict:
    def sysctl(key):
        try:
            return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {
        "os": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "processor": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": sysctl("hw.memsize"), "pid": os.getpid(),
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "caffeinate_assertions": os.environ.get("INFERENCE_CAFFEINATE_ASSERTIONS"),
        "swap_usage": sysctl("vm.swapusage"),
        **power_thermal_snapshot(),
    }


def resource_snapshot() -> dict:
    try:
        swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        swap = None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"swap_usage": swap,
            "process_max_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
            **power_thermal_snapshot()}
