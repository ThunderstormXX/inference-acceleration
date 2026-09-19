"""Sample existing macOS telemetry without GPU work or administrator access."""

from __future__ import annotations

import ctypes
import json
import plistlib
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


class HostDiagnostics:
    THERMAL_NAMES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}

    @staticmethod
    def command(*arguments: str) -> dict:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}

    @classmethod
    def thermal_state(cls) -> dict:
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        send_object = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc))
        send_integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc))
        info = send_object(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
        state = send_integer(info, objc.sel_registerName(b"thermalState"))
        return {"value": state, "name": cls.THERMAL_NAMES.get(state, "unknown"),
                "source": "NSProcessInfo.thermalState",
                "limitation": "System thermal category, not a GPU temperature or measured GPU frequency."}

    @classmethod
    def registry(cls, name: str) -> list[dict]:
        result = cls.command("ioreg", "-r", "-c", name, "-a")
        if result["returncode"]:
            raise RuntimeError(result["stderr"])
        return plistlib.loads(result["stdout"].encode())

    @classmethod
    def snapshot(cls) -> dict:
        snapshot = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "thermal": cls.thermal_state()}
        batteries = []
        for battery in cls.registry("AppleSmartBattery"):
            keep = ("ExternalConnected", "AppleRawExternalConnected", "ExternalChargeCapable",
                    "IsCharging", "CurrentCapacity", "Temperature", "InstantAmperage",
                    "Amperage", "Voltage")
            entry = {key: battery.get(key) for key in keep}
            entry["adapter_watts"] = battery.get("AdapterDetails", {}).get("Watts")
            current, voltage = entry.get("InstantAmperage"), entry.get("Voltage")
            if isinstance(current, int) and isinstance(voltage, int):
                if current > 2 ** 63:
                    current -= 2 ** 64
                entry["instant_battery_power_w_estimate"] = current * voltage / 1_000_000
            # Do not export battery serial numbers, device IDs or adapter IDs.
            batteries.append(entry)
        snapshot["batteries"] = batteries
        snapshot["gpu"] = [{"performance_statistics": device.get("PerformanceStatistics", {})}
                           for device in cls.registry("IOAccelerator")]
        snapshot["vm_stat"] = cls.command("vm_stat")
        snapshot["vm_counters"] = {key: int(value) for key, value in
                                   re.findall(r'^([^:\n]+):\s+(\d+)\.',
                                              snapshot["vm_stat"]["stdout"], re.MULTILINE)}
        snapshot["swap"] = cls.command("sysctl", "vm.swapusage")
        snapshot["power_source"] = cls.command("pmset", "-g", "batt")
        process_result = cls.command("ps", "-Ao", "pid,pcpu,pmem,rss,comm")
        processes = []
        if process_result["returncode"] == 0:
            for row in process_result["stdout"].splitlines()[1:]:
                fields = row.split(None, 4)
                if len(fields) == 5:
                    processes.append({"pid": int(fields[0]), "cpu_percent": float(fields[1]),
                                      "memory_percent": float(fields[2]), "rss_kib": int(fields[3]),
                                      "name": Path(fields[4]).name})
        snapshot["top_cpu_processes"] = sorted(processes, key=lambda item: item["cpu_percent"],
                                               reverse=True)[:10]
        return snapshot

    def run(self, output_directory: Path, samples: int = 3, interval: float = 5) -> Path:
        if samples < 1 or interval < 0 or interval > 60:
            raise ValueError("Positive sample count and interval in [0, 60] required")
        started = datetime.now(timezone.utc)
        snapshots = []
        for index in range(samples):
            snapshots.append(self.snapshot())
            print(json.dumps({key: snapshots[-1][key] for key in
                              ("timestamp_utc", "thermal", "batteries", "top_cpu_processes")}))
            if index + 1 < samples:
                time.sleep(interval)
        first, last = snapshots[0], snapshots[-1]
        deltas = {key: last["vm_counters"].get(key, 0) - first["vm_counters"].get(key, 0)
                  for key in ("Swapins", "Swapouts", "Pageins", "Pageouts", "Decompressions", "Compressions")}
        report = {"started_utc": started.isoformat(), "read_only": True, "gpu_compute": False,
                  "no_sudo": True, "samples": snapshots, "vm_counter_deltas": deltas,
                  "power_settings": self.command("pmset", "-g", "custom"),
                  "frequency_measurement": "Unavailable: powermetrics requires root; sudo was not used.",
                  "interpretation_limit": "Correlation only. This does not establish the cause of benchmark drift."}
        output_directory.mkdir(parents=True, exist_ok=True)
        output = output_directory / f"{started.strftime('%Y%m%dT%H%M%SZ')}-host-diagnostics.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Results: {output}")
        return output
