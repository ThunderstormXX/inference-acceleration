"""Read process priority and CPU QoS accounting without altering task policy."""

import argparse
import ctypes
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


class RusageInfoV3(ctypes.Structure):
    """Public macOS SDK sys/resource.h layout, RUSAGE_INFO_V3."""

    NAMES = ("user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size "
             "resident_size phys_footprint proc_start_abstime proc_exit_abstime "
             "child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups "
             "child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten "
             "cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background "
             "cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated "
             "cpu_time_qos_user_interactive billed_system_time serviced_system_time").split()
    _fields_ = [("uuid", ctypes.c_uint8 * 16), *[(name, ctypes.c_uint64) for name in NAMES]]


class ProcessPolicySnapshot:
    @staticmethod
    def command(*args):
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return {"command": list(args), "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}

    @staticmethod
    def accounting(pid):
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        library.proc_pid_rusage.restype = ctypes.c_int
        usage = RusageInfoV3()
        result = library.proc_pid_rusage(pid, 3, ctypes.byref(usage))
        if result != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        return {name: getattr(usage, name) for name in usage.NAMES if name.startswith("cpu_time_qos_")}

    def run(self, pid):
        started = datetime.now(timezone.utc)
        before = self.accounting(pid)
        report = {"started_utc": started.isoformat(), "pid": pid,
                  "priority": self.command("ps", "-p", str(pid), "-o", "pid,ppid,state,pri,nice,flags,wq,wqb,wqr,etime,comm"),
                  "launch_services": self.command("lsappinfo", "find", f"pid={pid}"),
                  "sleep_assertions": self.command("pmset", "-g", "assertions")}
        time.sleep(3)
        after = self.accounting(pid)
        delta = {key: after[key] - before[key] for key in before}
        total = sum(delta.values())
        report.update({"cpu_qos_before": before, "cpu_qos_after": after, "cpu_qos_delta": delta,
                       "cpu_qos_fraction": {key: value / total if total else None for key, value in delta.items()},
                       "read_only": True, "no_sudo": True,
                       "limitations": "CPU QoS accounting is not the GPU task policy. No LaunchServices match does not prove App Nap is disabled. No priority or power policy was changed."})
        directory = Path(__file__).resolve().parents[2] / "artifacts/reports"
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"{started.strftime('%Y%m%dT%H%M%SZ')}-process-policy.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print(f"Results: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", type=int)
    args = parser.parse_args()
    ProcessPolicySnapshot().run(args.pid)
