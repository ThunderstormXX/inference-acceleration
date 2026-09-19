"""Audit archived benchmark measurements without loading a model or GPU runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from inference_lab.core.config import ROOT
from inference_lab.core.io import write_json
from .metrics import DISPERSION_FIELDS, aggregate


class FinalSeriesValidator:
    """Select the latest complete run per engine and verify its raw evidence."""

    BACKENDS = ("transformers", "mlx", "mlx-vlm", "vllm")
    HASH_FIELDS = ("dataset_sha256", "model_manifest_sha256", "prompt_tokens_sha256")

    def __init__(self, runs: Path = ROOT / "artifacts/runs", label: str = "awake",
                 output: Path | None = None):
        self.runs = Path(runs).resolve()
        self.label = label
        self.output = Path(output or ROOT / "artifacts/reports/final-validation.json").resolve()

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    @staticmethod
    def _token_hash(tokens: list) -> str:
        # This is the exact serialization used by PromptSample and BenchmarkRunner.
        return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()

    @staticmethod
    def _read(path: Path) -> tuple[bytes, str]:
        payload = path.read_bytes()
        return payload, hashlib.sha256(payload).hexdigest()

    def _select(self) -> tuple[dict, list[str]]:
        selected, errors = {}, []
        for path in sorted(self.runs.glob("*/summary.json")):
            try:
                payload, checksum = self._read(path)
                summary = json.loads(payload)
                config = summary.get("config", {})
                backend = config.get("backend")
                if (backend not in self.BACKENDS or config.get("label") != self.label
                        or summary.get("status") != "completed"
                        or config.get("count") != 100 or summary.get("completed_samples") != 100
                        or (summary.get("environment") or {}).get("caffeinate_assertions") != "di"):
                    continue
                # Runner uses sortable UTC timestamps; the path breaks timestamp ties.
                key = (summary.get("started_at") or path.parent.name, str(path))
                if backend not in selected or key > selected[backend]["key"]:
                    selected[backend] = {"key": key, "path": path, "summary": summary, "sha256": checksum}
            except (OSError, ValueError, TypeError, AttributeError) as error:
                errors.append(f"Cannot read {path}: {error}")
        missing = [backend for backend in self.BACKENDS if backend not in selected]
        if missing:
            errors.append(
                f"Incomplete series: no completed 100/100 run with label={self.label!r} "
                f"and caffeinate_assertions='di' for {', '.join(missing)}"
            )
        return selected, errors

    def _compare_metrics(self, actual: dict, expected: dict, prefix: str = "metrics") -> None:
        self._require(isinstance(actual, dict), f"{prefix} must be an object")
        for key, value in expected.items():
            location = f"{prefix}.{key}"
            # Historical summaries predate dispersion statistics. Their original
            # aggregates remain mandatory; verify every new statistic if stored.
            if (key not in actual and key in DISPERSION_FIELDS
                    and prefix in ("metrics.prefill", "metrics.decode")):
                continue
            self._require(key in actual, f"Missing {location}")
            if isinstance(value, dict):
                self._compare_metrics(actual[key], value, location)
            elif value is None:
                self._require(actual[key] is None,
                              f"{location} differs: stored={actual[key]!r}, recalculated=None")
            elif type(value) is int:
                self._require(type(actual[key]) is int and actual[key] == value,
                              f"{location} differs: stored={actual[key]!r}, recalculated={value!r}")
            else:
                number = actual[key]
                self._require(type(number) in (int, float) and math.isfinite(number)
                              and math.isclose(number, value, rel_tol=1e-12, abs_tol=1e-12),
                              f"{location} differs: stored={number!r}, recalculated={value!r}")

    def _audit_run(self, candidate: dict) -> dict:
        path, summary = candidate["path"], candidate["summary"]
        config = summary["config"]
        output_count = config.get("max_new_tokens")
        self._require(type(output_count) is int and output_count >= 2,
                      "max_new_tokens must be an integer >= 2")
        prompts_path, samples_path = path.parent / "prompts.json", path.parent / "samples.jsonl"
        prompts_payload, prompts_file_hash = self._read(prompts_path)
        samples_payload, samples_file_hash = self._read(samples_path)
        prompts = json.loads(prompts_payload)
        lines = samples_payload.decode("utf-8").splitlines()
        self._require(len(lines) == 100 and all(line.strip() for line in lines),
                      f"samples.jsonl must contain exactly 100 nonempty rows, found {len(lines)}")
        rows = [json.loads(line) for line in lines]
        self._require(isinstance(prompts, list) and len(prompts) == 100,
                      "prompts.json must contain exactly 100 prompts")
        prompt_tokens = []
        for index, (prompt, row) in enumerate(zip(prompts, rows)):
            self._require(isinstance(prompt, dict) and isinstance(row, dict), f"Row {index} must be an object")
            self._require(type(prompt.get("index")) is int and prompt["index"] == index,
                          f"Prompt indices must be exactly 0..99 in order; mismatch at {index}")
            self._require(type(row.get("index")) is int and row["index"] == index,
                          f"Sample indices must be exactly 0..99 in order; mismatch at {index}")
            tokens = prompt.get("prompt_tokens")
            self._require(isinstance(tokens, list) and len(tokens) > 0
                          and all(type(token) is int and token >= 0 for token in tokens),
                          f"Prompt {index} has invalid token IDs")
            self._require(type(prompt.get("original_prompt_tokens")) is int
                          and prompt["original_prompt_tokens"] == len(tokens),
                          f"Prompt {index}: original_prompt_tokens differs from saved token count")
            self._require(type(row.get("prompt_tokens")) is int and row["prompt_tokens"] == len(tokens),
                          f"Sample {index}: prompt-token count differs from prompts.json")
            self._require(row.get("prompt_token_sha256") == self._token_hash(tokens),
                          f"Sample {index}: prompt-token SHA256 differs from prompts.json")
            generated = row.get("generated_token_ids")
            self._require(isinstance(generated, list) and len(generated) == output_count
                          and all(type(token) is int and token >= 0 for token in generated),
                          f"Sample {index}: expected {output_count} nonnegative output token IDs")
            self._require(type(row.get("generated_tokens")) is int and row["generated_tokens"] == output_count,
                          f"Sample {index}: generated_tokens differs from config")
            self._require(type(row.get("decode_tokens")) is int and row["decode_tokens"] == output_count - 1,
                          f"Sample {index}: decode_tokens must be {output_count - 1}, excluding the first token")
            for phase in ("prefill", "decode"):
                duration = row.get(f"{phase}_seconds")
                self._require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0,
                              f"Sample {index}: {phase}_seconds must be positive and finite")
            memory = row.get("peak_memory_gb", 0)
            self._require(type(memory) in (int, float) and math.isfinite(memory) and memory >= 0,
                          f"Sample {index}: peak_memory_gb must be nonnegative and finite")
            prompt_tokens.append(tokens)
        prompt_hash = self._token_hash(prompt_tokens)
        self._require(summary.get("prompt_tokens_sha256") == prompt_hash,
                      "Aggregate prompt-token SHA256 differs from prompts.json")
        recalculated = aggregate(rows)
        self._compare_metrics(summary.get("metrics"), recalculated)
        return {
            "status": "passed", "backend": config["backend"], "label": config["label"],
            "run_directory": str(path.parent), "summary_path": str(path),
            "summary_file_sha256": candidate["sha256"],
            "prompts_path": str(prompts_path), "prompts_file_sha256": prompts_file_hash,
            "samples_path": str(samples_path), "samples_file_sha256": samples_file_hash,
            "samples": len(rows), "prompts": len(prompts),
            "indices": {"first": rows[0]["index"], "last": rows[-1]["index"],
                        "unique": len({row["index"] for row in rows})},
            "max_new_tokens": output_count, "decode_tokens_per_sample": output_count - 1,
            "generated_token_ids_checked": sum(len(row["generated_token_ids"]) for row in rows),
            **{key: summary[key] for key in self.HASH_FIELDS},
            "caffeinate_assertions": summary["environment"]["caffeinate_assertions"],
            "user_initiated": config.get("user_initiated", False),
            "wired_memory": config.get("wired_memory", False),
            "recalculated_metrics": recalculated,
        }

    def run(self) -> dict:
        selected, errors = self._select()
        common, audits = {}, []
        for field in (*self.HASH_FIELDS, "max_new_tokens", "prompt_mode"):
            values = {backend: candidate["summary"].get(field) if field in self.HASH_FIELDS
                      else candidate["summary"]["config"].get(field)
                      for backend, candidate in selected.items()}
            valid = bool(values)
            if field in self.HASH_FIELDS:
                valid = valid and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                                      for value in values.values())
            else:
                valid = valid and all(value is not None for value in values.values())
            if valid and all(value == next(iter(values.values())) for value in values.values()):
                common[field] = next(iter(values.values()))
            elif values:
                errors.append(f"Series requires matching valid {field}: {values!r}")
        for backend in self.BACKENDS:
            if backend not in selected:
                continue
            candidate = selected[backend]
            try:
                audits.append(self._audit_run(candidate))
            except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
                message = f"{candidate['path']}: {error}"
                errors.append(message)
                audits.append({"status": "failed", "backend": backend,
                               "run_directory": str(candidate["path"].parent),
                               "summary_path": str(candidate["path"]), "error": str(error)})
        result = {
            "schema_version": 1, "status": "failed" if errors else "passed",
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "runs_directory": str(self.runs), "label": self.label,
            "required_backends": list(self.BACKENDS), "selected_backends": list(selected),
            "selection": "Latest completed 100/100 run per backend with matching label and caffeinate_assertions=di",
            "common": common, "runs": audits, "errors": errors,
            "scope": "Archived sample, prompt and summary integrity; no model loading or answer-quality evaluation.",
            "metric_tolerance": {"relative": 1e-12, "absolute": 1e-12},
        }
        write_json(self.output, result)
        print(f"Final series validation: {result['status']} -> {self.output}", flush=True)
        for error in errors:
            print(f"ERROR: {error}", flush=True)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts/runs")
    parser.add_argument("--label", default="awake")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reports/final-validation.json")
    args = parser.parse_args()
    result = FinalSeriesValidator(args.runs, args.label, args.output).run()
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
