"""Resumable paired MTP series: alternating process order and immutable raw evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
from importlib.metadata import PackageNotFoundError, distribution, version
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import tempfile

from inference_lab.core.config import ROOT, BenchmarkConfig
from inference_lab.core.io import sha256_file
from .metrics import aggregate
from .mtp import MTPConfig
from .speculative_report import SpeculativeReport
from .validation import FinalSeriesValidator


RAW_FILES = ("summary.json", "prompts.json", "samples.jsonl")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value):
    """Replace the state file atomically and fsync it before another child starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".series-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class MTPSeriesConfig:
    count: int = 100
    max_new_tokens: int = 2048
    chunk_size: int = 5
    block_size: int = 3
    prefill_step_size: int = 512
    label: str = "mtp-long2048"
    model_path: str = BenchmarkConfig.model_path
    draft_path: str = MTPConfig.draft_path
    dataset_path: str = BenchmarkConfig.dataset_path
    trace_generation: bool = True
    min_battery_percent: int = 20

    def __post_init__(self):
        BenchmarkConfig("mlx-vlm", count=self.count, max_new_tokens=self.max_new_tokens,
                        prefill_step_size=self.prefill_step_size, trace_generation=self.trace_generation)
        if type(self.chunk_size) is not int or not 1 <= self.chunk_size <= 100:
            raise ValueError("chunk_size must be an integer in 1..100")
        if type(self.block_size) is not int or not 2 <= self.block_size <= 5:
            raise ValueError("MTP block_size must be an integer in 2..5")
        if type(self.min_battery_percent) is not int or not 0 <= self.min_battery_percent <= 100:
            raise ValueError("min_battery_percent must be an integer in 0..100")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.label):
            raise ValueError("label may contain ASCII letters, digits, - and _ only")


class MTPSeriesRunner:
    """Run A0 B0 | B1 A1 | A2 B2 ... and merge only fully verified coverage."""

    def __init__(self, config: MTPSeriesConfig, output: Path | None = None, resume: bool = False):
        self.config = config
        self.output = Path(output or ROOT / "artifacts/series" / config.label).resolve()
        self.resume = resume
        self.manifest_path = self.output / "manifest.json"
        self.manifest = None

    def plan(self):
        blocks = []
        for chunk, start in enumerate(range(0, self.config.count, self.config.chunk_size)):
            roles = ("baseline", "mtp") if chunk % 2 == 0 else ("mtp", "baseline")
            for role in roles:
                blocks.append({"id": f"{chunk:03d}-{role}", "chunk_index": chunk, "role": role,
                               "start_index": start, "count": min(self.config.chunk_size, self.config.count - start)})
        return blocks

    def _worker_config(self, block):
        c = self.config
        label = (f"mtp-baseline-{c.label}-c{block['chunk_index']:03d}" if block["role"] == "baseline"
                 else f"mtp-k{c.block_size}-{c.label}-c{block['chunk_index']:03d}")
        common = dict(count=block["count"], start_index=block["start_index"], max_new_tokens=c.max_new_tokens,
                      prefill_step_size=c.prefill_step_size, warmup=1, label=label, wired_memory=True,
                      trace_generation=c.trace_generation, model_path=str(Path(c.model_path).resolve()),
                      dataset_path=str(Path(c.dataset_path).resolve()))
        if block["role"] == "baseline":
            return BenchmarkConfig("mlx-vlm", **common)
        return MTPConfig("mlx-mtp", draft_path=str(Path(c.draft_path).resolve()), block_size=c.block_size, **common)

    def command(self, block):
        c = self._worker_config(block)
        command = ["bash", str(ROOT / "scripts/benchmark/mtp.sh"), "--mode", block["role"],
                   "--start-index", str(c.start_index), "--count", str(c.count),
                   "--max-new-tokens", str(c.max_new_tokens), "--warmup", "1",
                   "--prefill-step-size", str(c.prefill_step_size), "--model-path", c.model_path,
                   "--dataset-path", c.dataset_path, "--label", c.label]
        if block["role"] == "mtp":
            command += ["--draft-path", c.draft_path, "--block-size", str(c.block_size)]
        if c.trace_generation:
            command.append("--trace-generation")
        return command

    def _fingerprint(self):
        # Freeze measurement/validation code, not unrelated README/export/render edits.
        relevant = ["src/inference_lab/data/prompts.py", "scripts/_common.sh",
                    "scripts/benchmark/_bootstrap.py", "requirements-mac.lock"]
        relevant += [f"src/inference_lab/benchmarking/{name}.py" for name in
                     ("runner", "metrics", "mtp", "mtp_series", "speculative_report", "validation", "recheck_report")]
        relevant += [f"src/inference_lab/visualization/{name}.py" for name in ("recording", "trace")]
        relevant += [f"scripts/benchmark/{name}.{extension}" for name in ("mtp", "mtp_series")
                     for extension in ("sh", "py")]
        files = sorted({*(ROOT / name for name in relevant), *ROOT.glob("src/inference_lab/core/*.py"),
                        *ROOT.glob("src/inference_lab/backends/**/*.py")})
        project = {str(path.relative_to(ROOT)): sha256_file(path) for path in files}
        packages = {}
        for name in ("mlx", "mlx-metal", "mlx-lm", "mlx-vlm", "transformers", "huggingface-hub"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        runtime_sources = {}
        for package in ("mlx-vlm", "mlx-lm"):
            dist = distribution(package)
            for entry in dist.files or []:
                if str(entry).endswith(".py") and ".." not in Path(entry).parts:
                    path = Path(dist.locate_file(entry))
                    runtime_sources[f"{package}/{entry}"] = sha256_file(path)
        return {"project_files": project,
                "runtime": {"python": platform.python_version(), "machine": platform.machine(),
                            "packages": packages, "source_files": runtime_sources},
                "inputs": {"dataset_sha256": sha256_file(Path(self.config.dataset_path)),
                           "model_manifest_sha256": sha256_file(Path(self.config.model_path) / "download-manifest.json"),
                           "draft_manifest_sha256": sha256_file(Path(self.config.draft_path) / "download-manifest.json")}}

    def _save(self):
        self.manifest["updated_at"] = _now()
        _write_json(self.manifest_path, self.manifest)

    def _battery(self):
        raw = subprocess.check_output(["pmset", "-g", "batt"], text=True, stderr=subprocess.PIPE, timeout=5)
        charge = re.search(r"(\d+)%;", raw)
        if charge is None:
            raise ValueError("Cannot determine battery percentage; series remains resumable")
        percent = int(charge.group(1))
        discharging = bool(re.search(r"\bdischarging\b", raw, re.I))
        return {"checked_at": _now(), "percent": percent, "discharging": discharging,
                "raw": raw.strip(), "stop": discharging and percent < self.config.min_battery_percent}

    @staticmethod
    def _files(directory):
        return {name: {"path": str(directory / name), "sha256": sha256_file(directory / name)} for name in RAW_FILES}

    @staticmethod
    def _verify_files(files):
        if set(files) != set(RAW_FILES):
            raise ValueError("Missing raw-file evidence for a completed block")
        for name, evidence in files.items():
            path = Path(evidence["path"])
            if path.name != name or sha256_file(path) != evidence["sha256"]:
                raise ValueError(f"Recorded source hash mismatch: {path}")

    def _validate_block(self, block, attempt):
        directory = Path(attempt["run_directory"]).resolve()
        if attempt.get("command") != self.command(block):
            raise ValueError("Recorded worker command differs from the planned protocol")
        if attempt.get("files"):
            self._verify_files(attempt["files"])
            if any(Path(record["path"]).parent != directory for record in attempt["files"].values()):
                raise ValueError("Recorded files do not belong to the selected run directory")
        role = "baseline" if block["role"] == "baseline" else "speculative"
        loader = SpeculativeReport(directory, [])
        loaded = loader._load(directory, role, {})
        summary = loaded["summary"]
        expected = asdict(self._worker_config(block))
        if summary["config"] != expected:
            raise ValueError(f"Block {block['id']}: worker config differs from suite protocol")
        indices = list(range(block["start_index"], block["start_index"] + block["count"]))
        if [row["index"] for row in loaded["rows"]] != indices:
            raise ValueError(f"Block {block['id']}: global indices do not exactly cover its requested slice")
        sources = self.manifest["sources"]
        for key in ("dataset_sha256", "model_manifest_sha256"):
            if summary[key] != sources["inputs"][key]:
                raise ValueError(f"Block {block['id']}: {key} differs from the frozen input")
        if block["role"] == "mtp" and summary["backend"].get("draft_manifest_sha256") != sources["inputs"]["draft_manifest_sha256"]:
            raise ValueError("MTP draft manifest differs from the frozen input")
        environment = summary["environment"]
        if environment.get("python") != sources["runtime"]["python"]:
            raise ValueError("Worker Python version differs from series runtime")
        reported = {key.lower().replace("_", "-"): value for key, value in environment.get("packages", {}).items()}
        for name, expected_version in sources["runtime"]["packages"].items():
            if reported.get(name) != expected_version:
                raise ValueError(f"Worker package version changed: {name}")
        FinalSeriesValidator._compare_metrics(self, summary.get("metrics"), aggregate(loaded["rows"]))
        if self.config.trace_generation:
            from inference_lab.visualization.recording import validate_generation_trace
            for row in loaded["rows"]:
                validate_generation_trace(row)
        attempt["files"] = self._files(directory)
        return loaded

    # Reuse the archive validator's recursive metric comparison without creating files.
    _require = staticmethod(FinalSeriesValidator._require)
    _compare_metrics = FinalSeriesValidator._compare_metrics

    def _execute(self, block, attempt):
        """Stream child output to a unique append-only log; persist its Results path immediately."""
        log_path = Path(attempt["log"])
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        for key in ("INFERENCE_PYTHON", "PYTHONPATH", "VIRTUAL_ENV"):
            env.pop(key, None)
        with log_path.open("x") as log:
            process = subprocess.Popen(attempt["command"], cwd=ROOT, env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       bufsize=1, start_new_session=True)
            attempt["pid"] = process.pid
            self._save()
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    if line.startswith("Results: "):
                        path = Path(line[len("Results: "):].strip()).resolve()
                        if not path.is_relative_to((ROOT / "artifacts/runs").resolve()):
                            raise ValueError("Worker returned an unexpected Results directory")
                        if attempt.get("run_directory") not in (None, str(path)):
                            raise ValueError("Worker returned multiple Results directories")
                        attempt["run_directory"] = str(path)
                        self._save()
                return process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                raise
            finally:
                process.stdout.close()
                log.flush()
                os.fsync(log.fileno())

    def _recover(self, block):
        if block["status"] == "completed":
            if not block["attempts"] or block["attempts"][-1].get("status") != "completed" or not block["attempts"][-1].get("files"):
                raise ValueError("Completed block is missing its completed attempt/hash evidence")
            self._validate_block(block, block["attempts"][-1])
            return True
        if not block["attempts"]:
            return False
        previous = block["attempts"][-1]
        if previous["status"] == "running":
            if previous.get("pid"):
                try:
                    os.kill(previous["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f"Previous child PID {previous['pid']} is still alive; refusing duplicate GPU work")
            if previous.get("run_directory"):
                try:
                    self._validate_block(block, previous)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    previous["recovery_error"] = str(error)
                else:
                    previous.update(status="completed", recovered=True, finished_at=_now())
                    block["status"] = "completed"
                    self._save()
                    return True
            previous.update(status="interrupted", finished_at=_now())
            block["status"] = "pending"
            self._save()
        return False

    def _run_block(self, block):
        if self._fingerprint() != self.manifest["sources"]:
            raise ValueError("Measurement source hashes/runtime/inputs changed before next block")
        power = self._battery()
        self.manifest.setdefault("power_checks", []).append({"before_block": block["id"], **power})
        self._save()
        if power["stop"]:
            self.manifest.update(status="paused_battery", stop_reason=f"Discharging at {power['percent']}%, below {self.config.min_battery_percent}%")
            self._save()
            return False
        attempt_number = len(block["attempts"]) + 1
        log_dir = self.output / "logs"
        log_dir.mkdir(exist_ok=True)
        attempt = {"attempt": attempt_number, "status": "running", "started_at": _now(),
                   "command": self.command(block), "log": str(log_dir / f"{block['id']}-attempt-{attempt_number:03d}.log")}
        block["attempts"].append(attempt)
        block["status"] = "running"
        self._save()
        try:
            attempt["exit_code"] = self._execute(block, attempt)
            if attempt["exit_code"] != 0:
                raise RuntimeError(f"Child failed with exit code {attempt['exit_code']}; see {attempt['log']}")
            if not attempt.get("run_directory"):
                raise ValueError("Successful worker did not report its Results directory")
            self._validate_block(block, attempt)
            if self._fingerprint() != self.manifest["sources"]:
                raise ValueError("Measurement source hashes/runtime/inputs changed during block")
        except BaseException as error:
            attempt.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           finished_at=_now(), error=f"{type(error).__name__}: {error}")
            block["status"] = attempt["status"]
            self._save()
            raise
        attempt.update(status="completed", finished_at=_now())
        block["status"] = "completed"
        self._save()
        return True

    def _merge(self):
        blocks = self.manifest["blocks"]
        if any(block["status"] != "completed" for block in blocks):
            raise ValueError("Cannot merge a partial series")
        loaded = {block["id"]: self._validate_block(block, block["attempts"][-1]) for block in blocks}
        pair_reports = []
        for chunk in sorted({block["chunk_index"] for block in blocks}):
            pair = {block["role"]: block for block in blocks if block["chunk_index"] == chunk}
            baseline = Path(pair["baseline"]["attempts"][-1]["run_directory"])
            mtp = Path(pair["mtp"]["attempts"][-1]["run_directory"])
            report = SpeculativeReport(baseline, [mtp]).analyze()
            if report["status"] != "completed":
                raise ValueError(f"Pair {chunk} failed validation: {report['errors']}")
            pair_reports.append({"chunk_index": chunk, "report": report})
        merges = self.manifest.setdefault("merge_attempts", [])
        number = len(merges) + 1
        merge_root = self.output / "merged" / f"attempt-{number:03d}"
        while merge_root.exists():
            number += 1
            merge_root = self.output / "merged" / f"attempt-{number:03d}"
        merge_root.mkdir(parents=True, exist_ok=False)
        merge_attempt = {"status": "running", "directory": str(merge_root), "started_at": _now()}
        merges.append(merge_attempt)
        self._save()
        merged = {}
        for role in ("baseline", "mtp"):
            selected = sorted([block for block in blocks if block["role"] == role], key=lambda block: block["start_index"])
            rows = [row for block in selected for row in loaded[block["id"]]["rows"]]
            prompts = [prompt for block in selected for prompt in loaded[block["id"]]["prompts"]]
            if [row["index"] for row in rows] != list(range(self.config.count)) or [p["index"] for p in prompts] != list(range(self.config.count)):
                raise ValueError("Merged coverage must contain each global index exactly once, in order")
            originals = [loaded[block["id"]]["summary"] for block in selected]
            first = originals[0]
            config = deepcopy(first["config"])
            config.update(count=self.config.count, start_index=0,
                          label=f"mtp-baseline-{self.config.label}-merged" if role == "baseline" else f"mtp-k{self.config.block_size}-{self.config.label}-merged")
            sources = [{**{key: block[key] for key in ("id", "chunk_index", "role", "start_index", "count")},
                        **{key: block["attempts"][-1][key] for key in ("run_directory", "attempt", "command", "files")}}
                       for block in selected]
            directory = merge_root / role
            directory.mkdir()
            _write_json(directory / "prompts.json", prompts)
            with (directory / "samples.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            summary = {"status": "completed", "completed_samples": len(rows), "config": config,
                       "started_at": min(item["started_at"] for item in originals), "merged_at": _now(),
                       "run_directory": str(directory), "metrics": aggregate(rows),
                       "elapsed_seconds": sum(item.get("elapsed_seconds", 0) for item in originals),
                       "elapsed_seconds_scope": "Sum of source process elapsed times, including their individual loads and warmups; excludes gaps between processes",
                       "environment": {"python": first["environment"]["python"],
                                       "packages": first["environment"]["packages"], "caffeinate_assertions": "di",
                                       "scope": "Common runtime only; per-process power/thermal observations remain in source summaries"},
                       "backend": deepcopy(first["backend"]),
                       "model_manifest_sha256": first["model_manifest_sha256"], "dataset_sha256": first["dataset_sha256"],
                       "prompt_tokens_sha256": SpeculativeReport._hash([p["prompt_tokens"] for p in prompts]),
                       "series": {"label": self.config.label, "suite_manifest": str(self.manifest_path),
                                  "source_block_runs": sources, "execution_order": self.manifest["order"],
                                  "process_count": len(selected), "warmups_per_process": 1,
                                  "total_warmup_requests": len(selected), "model_load_count": len(selected),
                                  "methodology": "Separate process and one full-length warmup per chunk; merged statistics from measured rows, not a single model load."}}
            _write_json(directory / "summary.json", summary)
            merged[role] = str(directory)
        final_report = SpeculativeReport(Path(merged["baseline"]), [Path(merged["mtp"])]).analyze()
        if final_report["status"] != "completed":
            raise ValueError(f"Merged pair failed validation: {final_report['errors']}")
        _write_json(merge_root / "pair-reports.json", pair_reports)
        _write_json(merge_root / "comparison.json", final_report)
        evidence = {role: self._files(Path(directory)) for role, directory in merged.items()}
        merge_attempt.update(status="completed", finished_at=_now(), files=evidence)
        self.manifest.update(merged_runs=merged, merged_files=evidence,
                             comparison_path=str(merge_root / "comparison.json"), parity_status=final_report["parity_status"])
        self._save()

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "suite.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("This series is already running") from error
            sources = self._fingerprint()
            if self.manifest_path.exists():
                if not self.resume:
                    raise ValueError("Series already exists; use --resume to verify and continue")
                self.manifest = json.loads(self.manifest_path.read_text())
            else:
                self.manifest = {"schema_version": 1, "status": "running", "created_at": _now(),
                                 "config": asdict(self.config), "sources": sources,
                                 "order": [block["id"] for block in self.plan()],
                                 "blocks": [{**block, "status": "pending", "attempts": []} for block in self.plan()]}
                self._save()
            try:
                if self.manifest["config"] != asdict(self.config) or self.manifest["sources"] != sources:
                    raise ValueError("Resume source hashes/runtime/inputs or suite protocol changed")
                plan = self.plan()
                if self.manifest["order"] != [block["id"] for block in plan] or len(self.manifest["blocks"]) != len(plan):
                    raise ValueError("Recorded execution order differs from ABBA plan")
                for expected, block in zip(plan, self.manifest["blocks"]):
                    if any(block.get(key) != value for key, value in expected.items()):
                        raise ValueError("Recorded block protocol differs from its plan")
                    if self._recover(block):
                        continue
                    self.manifest["status"] = "running"
                    self.manifest.pop("error", None)
                    self._save()
                    if not self._run_block(block):
                        return self.manifest
                if self.manifest.get("merged_runs"):
                    for files in self.manifest["merged_files"].values():
                        self._verify_files(files)
                else:
                    self._merge()
                self.manifest.update(status="completed", finished_at=_now())
                self._save()
                return self.manifest
            except BaseException as error:
                self.manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                     error=f"{type(error).__name__}: {error}")
                self._save()
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--label", default="mtp-long2048")
    parser.add_argument("--model-path", default=BenchmarkConfig.model_path)
    parser.add_argument("--draft-path", default=MTPConfig.draft_path)
    parser.add_argument("--dataset-path", default=BenchmarkConfig.dataset_path)
    parser.add_argument("--trace-generation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-battery-percent", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    output, resume = args.pop("output"), args.pop("resume")
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Series received SIGTERM; active child is being stopped")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = MTPSeriesRunner(MTPSeriesConfig(**args), output, resume).run()
        print(f"Series {result['status']}: {Path(output or ROOT / 'artifacts/series' / args['label']).resolve() / 'manifest.json'}", flush=True)
        return 0 if result["status"] == "completed" else 2
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
