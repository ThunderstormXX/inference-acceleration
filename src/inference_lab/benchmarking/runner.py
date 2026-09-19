"""Sequential, restart-visible benchmarking with per-sample durable results."""
import fcntl
import hashlib
import json
import os
import time
import traceback
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from inference_lab.backends.base import create_backend
from inference_lab.core.config import ROOT, BenchmarkConfig
from inference_lab.core.activity import UserInitiatedActivity
from inference_lab.core.io import environment, resource_snapshot, sha256_file, write_json
from inference_lab.data.prompts import PromptDataset
from .metrics import aggregate, validate_measurement


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, backend_factory=None):
        self.config = config
        self._backend_factory = backend_factory

    def run(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        name = self.config.backend + (f"-{self.config.label}" if self.config.label else "")
        directory = ROOT / "artifacts/runs" / f"{stamp}-{name}"
        directory.mkdir(parents=True)
        activity = UserInitiatedActivity(
            self.config.user_initiated,
            reason=f"User-requested {self.config.backend} inference benchmark ({self.config.count} requests)",
        )
        status = {"status": "running", "started_at": stamp, "config": asdict(self.config),
                  "environment": environment(), "run_directory": str(directory),
                  "activity": activity.metadata()}
        rows = []
        start = time.perf_counter()
        lock_path = ROOT / "artifacts/gpu.lock"
        activity_scope = ExitStack()
        with lock_path.open("w") as lock:
            write_json(directory / "summary.json", status)
            print(f"Results: {directory}", flush=True)
            try:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("Another inference benchmark holds the GPU lock")
                activity_scope.enter_context(activity)
                status["activity"] = activity.metadata()
                status["dataset_sha256"] = sha256_file(Path(self.config.dataset_path))
                model_manifest = Path(self.config.model_path) / "download-manifest.json"
                if model_manifest.is_file():
                    status["model_manifest"] = json.loads(model_manifest.read_text())
                    status["model_manifest_sha256"] = sha256_file(model_manifest)
                backend = (self._backend_factory or create_backend)(self.config)
                if getattr(self.config, "trace_generation", False):
                    if not hasattr(backend, "trace_generation"):
                        raise ValueError("Generation tracing is supported only by the MLX/MLX-VLM/MTP adapters")
                    backend.trace_generation = True
                backend.load()
                status["backend"] = backend.metadata()
                samples = PromptDataset(self.config, backend.tokenizer).load()
                status["prompt_tokens_sha256"] = hashlib.sha256(
                    json.dumps([s.prompt_tokens for s in samples]).encode()).hexdigest()
                write_json(directory / "prompts.json", [asdict(s) for s in samples])
                write_json(directory / "summary.json", status)
                for i in range(self.config.warmup):
                    print(f"Warmup {i+1}/{self.config.warmup}", flush=True)
                    backend.measure(samples[0].prompt_tokens, self.config.max_new_tokens)
                with (directory / "samples.jsonl").open("w") as output:
                    for sample in samples:
                        measurement = backend.measure(sample.prompt_tokens, self.config.max_new_tokens)
                        validate_measurement(measurement)
                        if measurement["prompt_tokens"] != len(sample.prompt_tokens):
                            raise ValueError("Backend prompt-token count does not match input")
                        if measurement["decode_tokens"] != measurement["generated_tokens"] - 1:
                            raise ValueError("Decode must exclude the first token produced by prefill")
                        if measurement["generated_tokens"] != self.config.max_new_tokens:
                            raise ValueError("Backend output length does not match configured max_new_tokens")
                        generated = measurement.get("generated_token_ids")
                        if (not isinstance(generated, list)
                                or len(generated) != measurement["generated_tokens"]
                                or any(type(token) is not int or token < 0 for token in generated)):
                            raise ValueError("Backend must return one nonnegative integer ID per generated token")
                        if getattr(self.config, "trace_generation", False):
                            from inference_lab.visualization.recording import validate_generation_trace
                            validate_generation_trace(measurement)
                        measurement.update(index=sample.index, prompt_token_sha256=sample.token_sha256)
                        output.write(json.dumps(measurement, ensure_ascii=False) + "\n")
                        output.flush()
                        if getattr(self.config, "trace_generation", False):
                            os.fsync(output.fileno())
                        rows.append(measurement)
                        pp = measurement["prompt_tokens"] / measurement["prefill_seconds"]
                        tg = measurement["decode_tokens"] / measurement["decode_seconds"]
                        print(f"[{len(rows):3}/{len(samples)}] prefill={pp:.2f} tok/s decode={tg:.2f} tok/s", flush=True)
                status["status"] = "completed"
            except (Exception, KeyboardInterrupt) as error:
                status.update(status="failed", error_type=type(error).__name__, error=str(error))
                (directory / "error.txt").write_text(traceback.format_exc())
                print(f"FAILED: {type(error).__name__}: {error}", flush=True)
            finally:
                try:
                    status["elapsed_seconds"] = time.perf_counter() - start
                    status["resources_after"] = resource_snapshot()
                    status["metrics"] = aggregate(rows)
                    status["completed_samples"] = len(rows)
                finally:
                    try:
                        activity_scope.close()
                    except (Exception, KeyboardInterrupt) as error:
                        status.update(status="failed", error_type=type(error).__name__, error=str(error))
                        (directory / "error.txt").write_text(traceback.format_exc())
                    finally:
                        status["activity"] = activity.metadata()
                        write_json(directory / "summary.json", status)
        if status["status"] != "completed":
            raise RuntimeError(f"Benchmark failed; see {directory / 'error.txt'}")
        print(json.dumps(status["metrics"], indent=2), flush=True)
        return directory
