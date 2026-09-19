"""Publish reproducible, privacy-curated measurements from explicit paired runs.

Raw text, token arrays, traces, process inventories and host identities stay local.
Source SHA256 values always describe the original files, never sanitized copies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev, variance

from inference_lab.core.config import ROOT
from .metrics import aggregate
from .speculative_report import SpeculativeReport


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def object_digest(value) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def distribution(values: list[float]) -> dict:
    return {"count": len(values), "mean": mean(values), "median": median(values),
            "sample_stddev": stdev(values) if len(values) > 1 else None,
            "sample_variance": variance(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


class PublicationExporter:
    """Validate a completed pair and export only explicitly selected public fields."""

    PACKAGES = {"mlx", "mlx-metal", "mlx-lm", "mlx-vlm", "transformers", "torch",
                "vllm", "vllm-metal", "dflash", "kernels", "numpy", "safetensors",
                "huggingface-hub", "tokenizers"}
    BACKEND_FIELDS = ("framework", "device", "model_type", "weight_bits", "weight_group_size",
                      "wired_memory", "sampling", "ignore_eos", "batch_size",
                      "fresh_cache_per_request", "first_generated_token_phase", "text_only",
                      "target_framework", "algorithm", "upstream_repository", "upstream_commit",
                      "draft_repo_id", "draft_revision", "draft_manifest_sha256",
                      "draft_weight_bits", "draft_weight_group_size", "block_size",
                      "max_draft_tokens_per_round", "timed_text_detokenization",
                      "measurement_harness", "compute_runtime", "model_implementation",
                      "generation_timing_policy", "trace_generation")
    COUNTERS = ("drafted_tokens", "accepted_draft_tokens", "speculative_rounds",
                "emitted_draft_tokens", "emitted_target_tokens")

    def __init__(self, baseline: Path, speculative: Path, output: Path):
        self.baseline = Path(baseline).resolve()
        self.speculative = Path(speculative).resolve()
        output = Path(output).resolve()
        self.output = output.with_suffix("") if output.suffix in (".json", ".md") else output
        for directory in (self.baseline, self.speculative):
            if self.output.is_relative_to(directory):
                raise ValueError("Publication output must be outside original run directories")

    @staticmethod
    def _path(value) -> str:
        path = Path(value)
        if not path.is_absolute():
            return path.as_posix() if ".." not in path.parts else "external/" + path.name
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            # Preserve portable paths from the previous checkout after migration.
            for anchor in ("artifacts", "models", "data", ".venv", ".venv-speculative", "src"):
                if anchor in path.parts:
                    return Path(*path.parts[path.parts.index(anchor):]).as_posix()
            return "external/" + path.name

    @classmethod
    def _text(cls, value):
        """Defense in depth for retained descriptive strings; never retain home paths."""
        if not isinstance(value, str):
            return value
        return re.sub(r"/(?:Users|home|private|tmp)/[^\s\"'<>;,)]*", "[local-path]", value)

    @classmethod
    def _pick(cls, source, keys):
        return {key: cls._text(source[key]) for key in keys
                if key in source and isinstance(source[key], (str, int, float, bool, type(None)))}

    @classmethod
    def _versions(cls, source):
        return {key.lower().replace("_", "-"): cls._text(value)
                for key, value in (source or {}).items()
                if key.lower().replace("_", "-") in cls.PACKAGES and isinstance(value, str)}

    @classmethod
    def _conditions(cls, source):
        source = source or {}
        power = source.get("power_source") or {}
        raw = power.get("raw", "") if isinstance(power, dict) else ""
        source_match = re.search(r"""Now drawing from ['"]?(AC Power|Battery Power)['"]?""", raw)
        charge_match = re.search(r"(\d+)%;\s*([^;\n]+)", raw)
        thermal = source.get("thermal_state") or {}
        return {"caffeinate_assertions": source.get("caffeinate_assertions"),
                "power_source": {"source": "pmset -g batt", "kind": source_match[1] if source_match else None,
                                 "battery_percent": int(charge_match[1]) if charge_match else None,
                                 "battery_state": cls._text(charge_match[2]) if charge_match else None},
                "thermal_state": cls._pick(thermal, ("status", "value", "name", "state", "source", "limitation"))
                if isinstance(thermal, dict) else cls._text(thermal)}

    @classmethod
    def _files(cls, files, verify=False):
        result = {}
        for name in ("summary.json", "prompts.json", "samples.jsonl"):
            if name not in files:
                continue
            item = files[name]
            if verify and digest(Path(item["path"]).read_bytes()) != item["sha256"]:
                raise ValueError(f"Source file changed or hash mismatch: {name}")
            result[name] = {"path": cls._path(item["path"]), "sha256": item["sha256"]}
        return result

    @classmethod
    def _guard_policy(cls, value):
        if (not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1
                or value.get("clock_method") != "mach_continuous_time minus mach_absolute_time"
                or type(value.get("require_ac")) is not bool
                or type(value.get("min_battery_percent")) is not int
                or not 0 <= value["min_battery_percent"] <= 100
                or value.get("power_observation_scope") != "before/after child process only; intervening power changes are not observed"):
            raise ValueError("Invalid declared host guard policy")
        threshold = value.get("sleep_threshold_seconds")
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold != 1.0:
            raise ValueError("Unsupported host guard sleep threshold; schema 1 requires one second")
        return cls._pick(value, ("schema_version", "clock_method", "sleep_threshold_seconds", "require_ac",
                                 "min_battery_percent", "power_observation_scope"))

    @classmethod
    def _guard_observation(cls, observation, policy):
        from inference_lab.core.host_clock import MacSleepClock
        if (not isinstance(observation, dict) or observation.get("schema_version") != 1
                or observation.get("valid") is not True or observation.get("invalid_reasons") != []
                or observation.get("collection_errors", []) != []):
            raise ValueError("Missing or invalid source chunk host observation")
        assessment = MacSleepClock.assess(observation["clock_before"], observation["clock_after"],
                                          threshold_seconds=policy["sleep_threshold_seconds"])
        if assessment != observation.get("sleep_assessment") or assessment.get("sleep_detected") is not False:
            raise ValueError("Source chunk sleep evidence is inconsistent or reports sleep")
        result = {"schema_version": 1, "valid": True, "invalid_reasons": [],
                  "sleep_assessment": assessment, "raw_observation_object_sha256": object_digest(observation)}
        for key in ("clock_before", "clock_after"):
            result[key] = cls._pick(observation[key], ("schema_version", "absolute_ticks", "continuous_ticks",
                                                       "timebase_numer", "timebase_denom", "sampling_span_ticks"))
        for key in ("power_before", "power_after"):
            power = observation.get(key)
            if not isinstance(power, dict) or not isinstance(power.get("raw"), str):
                raise ValueError("Missing source chunk power evidence")
            parsed = cls._conditions({"power_source": {"raw": power["raw"]}})["power_source"]
            if parsed["kind"] not in ("AC Power", "Battery Power") or parsed["battery_percent"] is None:
                raise ValueError("Unparseable source chunk power evidence")
            percent = parsed["battery_percent"]
            discharging = bool(re.search(r"\bdischarging\b", power["raw"], re.I))
            on_ac = parsed["kind"] == "AC Power"
            stop = discharging and percent < policy["min_battery_percent"]
            if (type(power.get("percent")) is not int or power["percent"] != percent or not 0 <= percent <= 100
                    or any(type(power.get(name)) is not bool for name in ("on_ac", "discharging", "stop"))
                    or power["on_ac"] != on_ac or power["discharging"] != discharging or power["stop"] != stop
                    or not isinstance(power.get("checked_at"), str) or not power["checked_at"]):
                raise ValueError("Source chunk parsed power evidence is inconsistent")
            if stop or (policy["require_ac"] and (not on_ac or discharging)):
                raise ValueError("Source chunk violates declared AC/battery policy")
            result[key] = {**cls._pick(power, ("checked_at", "percent", "discharging", "on_ac", "stop")),
                           "source": "pmset -g batt", "power_kind": parsed["kind"],
                           "battery_state": parsed["battery_state"]}
        return result

    @classmethod
    def _guard_attempt(cls, manifest, block):
        matches = [item for item in manifest.get("blocks", []) if item.get("id") == block["id"]]
        if len(matches) != 1 or matches[0].get("status") != "completed":
            raise ValueError("Guarded chunk is absent or incomplete in suite manifest")
        source = matches[0]
        if any(source.get(key) != block.get(key) for key in ("chunk_index", "role", "start_index", "count")):
            raise ValueError("Guarded chunk protocol conflicts with suite manifest")
        attempts = [item for item in source.get("attempts", []) if item.get("attempt") == block.get("attempt")]
        if len(attempts) != 1:
            raise ValueError("Guarded source attempt is missing or ambiguous")
        attempt = attempts[0]
        if (attempt.get("status") != "completed" or attempt.get("run_directory") != block["run_directory"]
                or attempt.get("files") != block["files"]
                or attempt.get("host_observation") != block.get("host_observation")):
            raise ValueError("Guarded source attempt evidence differs from suite manifest")

    @classmethod
    def _series(cls, summary, merged_prompts, merged_rows):
        series = summary.get("series")
        if series is None:
            return None, None
        if not isinstance(series, dict) or not series:
            raise ValueError("Invalid merged-series provenance")
        blocks = series.get("source_block_runs")
        if not isinstance(blocks, list) or any(not isinstance(block, dict) for block in blocks):
            raise ValueError("Invalid source chunk list")
        result = cls._pick(series, ("label", "process_count", "warmups_per_process",
                                   "total_warmup_requests", "model_load_count", "methodology"))
        result["execution_order"] = [cls._text(item) for item in series.get("execution_order", [])]
        manifest_data = None
        if series.get("suite_manifest"):
            manifest = Path(series["suite_manifest"])
            if (manifest.parent / "invalidation.json").exists():
                raise ValueError("Series has an explicit invalidation.json; publication is excluded")
            manifest_raw = manifest.read_bytes()
            manifest_data = json.loads(manifest_raw)
            if not isinstance(manifest_data, dict):
                raise ValueError("Invalid suite manifest object")
            result["suite_manifest"] = {"path": cls._path(manifest), "sha256": digest(manifest_raw)}
        declared_policy = series.get("host_guard_policy")
        manifest_policy = (manifest_data or {}).get("host_guard_policy")
        if declared_policy is not None or manifest_policy is not None:
            if declared_policy != manifest_policy or manifest_data.get("status") != "completed":
                raise ValueError("Declared host guard policy requires a matching completed suite manifest")
            policy = cls._guard_policy(declared_policy)
            result["host_guard_policy"] = policy
            result["host_guard"] = {"status": "validated", "scope": "Sleep assessed across each child process; AC/battery observed only before and after it.",
                                     "power_observation_limitation": "Intervening power-source changes are not observed."}
        else:
            policy = None
            if (manifest_data or {}).get("schema_version", 1) >= 2:
                raise ValueError("New suite manifest is missing its declared host guard policy")
            if any("host_observation" in block for block in blocks):
                raise ValueError("Host observations exist without a declared guard policy")
            result["host_guard"] = {"status": "not_guarded", "scope": "Legacy series: no validated sleep/power guard evidence; caffeinate does not prove absence of sleep."}
        result["source_block_runs"] = []
        common, metadata_sources = None, []
        expected_role = {"mlx": "baseline", "mlx-vlm": "baseline", "mlx-dflash": "dflash", "mlx-mtp": "mtp"}[summary["config"]["backend"]]
        expected_prompts = {item["index"]: item for item in merged_prompts}
        expected_rows = {item["index"]: item for item in merged_rows}
        covered, block_ids, chunk_indices, directories = set(), set(), set(), set()
        for block in blocks:
            block_id, chunk_index = block.get("id"), block.get("chunk_index")
            directory = Path(block["run_directory"]).resolve()
            if (not isinstance(block_id, str) or not block_id or block_id in block_ids
                    or type(chunk_index) is not int or chunk_index < 0 or chunk_index in chunk_indices
                    or directory in directories):
                raise ValueError("Duplicate or invalid source chunk identity")
            if block.get("role") != expected_role:
                raise ValueError("Source chunk role conflicts with merged backend")
            block_ids.add(block_id)
            chunk_indices.add(chunk_index)
            directories.add(directory)
            item = cls._pick(block, ("id", "chunk_index", "role", "start_index", "count", "attempt"))
            item["run_directory"] = cls._path(block["run_directory"])
            if policy is not None:
                cls._guard_attempt(manifest_data, block)
                item["host_observation"] = cls._guard_observation(block.get("host_observation"), policy)
            source_data = {}
            for name in ("summary.json", "prompts.json", "samples.jsonl"):
                source = block["files"][name]
                path = Path(source["path"]).resolve()
                if path != directory / name:
                    raise ValueError("Source chunk file path conflicts with its run directory")
                raw = path.read_bytes()
                if digest(raw) != source["sha256"]:
                    raise ValueError(f"Source file changed or hash mismatch: {name}")
                source_data[name] = raw
            item["files"] = cls._files(block["files"])
            chunk = json.loads(source_data["summary.json"])
            prompts = json.loads(source_data["prompts.json"])
            rows = [json.loads(line) for line in source_data["samples.jsonl"].splitlines() if line.strip()]
            if (not isinstance(chunk, dict) or not isinstance(prompts, list)
                    or any(not isinstance(value, dict) for value in [*prompts, *rows])):
                raise ValueError("Invalid source chunk raw objects")
            count, start_index = block.get("count"), block.get("start_index")
            if type(count) is not int or count < 1 or type(start_index) is not int or start_index < 0:
                raise ValueError("Invalid source chunk range")
            chunk_config = chunk.get("config", {})
            if (chunk.get("status") != "completed" or chunk.get("completed_samples") != count
                    or chunk_config.get("count") != count or chunk_config.get("start_index", 0) != start_index):
                raise ValueError("Source chunk completion/count/range conflicts with provenance")
            if chunk_config.get("backend") != summary["config"]["backend"]:
                raise ValueError("Source chunk backend conflicts with merged role")
            for key in (*SpeculativeReport.PROTOCOL, "wired_memory", "trace_generation"):
                if key == "count":
                    continue
                default = False if key == "trace_generation" else None
                if chunk_config.get(key, default) != summary["config"].get(key, default):
                    raise ValueError(f"Source chunk protocol differs from merged run: {key}")
            if chunk.get("dataset_sha256") != summary.get("dataset_sha256"):
                raise ValueError("Source chunk dataset hash differs from merged run")
            if SpeculativeReport._versions(chunk) != SpeculativeReport._versions(summary):
                raise ValueError("Source chunk runtime versions differ from merged run")
            indices = list(range(start_index, start_index + count))
            if ([value.get("index") for value in prompts] != indices
                    or [value.get("index") for value in rows] != indices):
                raise ValueError("Source chunk raw indices/count differ from provenance")
            if covered.intersection(indices):
                raise ValueError("Duplicate source chunk prompt coverage")
            for index, prompt, row in zip(indices, prompts, rows):
                if index not in expected_rows:
                    raise ValueError("Extra source chunk prompt outside merged coverage")
                if prompt != expected_prompts[index] or row != expected_rows[index]:
                    raise ValueError(f"Merged prompt or full sample differs from verified source chunk at index {index}")
            covered.update(indices)
            if chunk.get("model_manifest_sha256") != summary.get("model_manifest_sha256"):
                raise ValueError("Source chunk target manifest hash differs from merged run")
            hardware = cls._runtime(chunk)["hardware"]
            model = cls._model(chunk)
            if common is None:
                common = {"hardware": hardware, "model": model,
                          "model_manifest": chunk.get("model_manifest") or {}}
            elif hardware != common["hardware"] or model != common["model"]:
                raise ValueError("Inconsistent hardware or target model metadata across source chunks")
            item.update(cls._pick(chunk, ("started_at", "elapsed_seconds")))
            item["elapsed_scope"] = "Original runner elapsed_seconds: includes model loading, warmup, measured requests and bookkeeping before final resource snapshot; not pooled phase time."
            item["conditions_before"] = cls._conditions(chunk.get("environment"))
            item["conditions_after"] = cls._conditions(chunk.get("resources_after"))
            item["conditions_scope"] = "Snapshots around this source process, not continuous telemetry or evidence of throttling."
            metadata_sources.append(item["files"]["summary.json"])
            result["source_block_runs"].append(item)
        if common is None:
            raise ValueError("Merged series requires at least one verified source chunk")
        if covered != set(expected_rows):
            raise ValueError("Missing source chunks for merged prompt coverage")
        for key in ("process_count", "model_load_count"):
            if key in series and series[key] != len(block_ids):
                raise ValueError(f"Source chunk count differs from series.{key}")
        order = series.get("execution_order")
        if (not isinstance(order, list) or any(not isinstance(value, str) for value in order)
                or len(set(order)) != len(order) or not block_ids.issubset(order)):
            raise ValueError("Missing or duplicate source chunk IDs in execution_order")
        result["lineage_validation"] = "Every merged prompt and full sample equals its hash-verified source object; exact index coverage, no duplicate/extra chunks."
        result["metadata_provenance"] = {
            "scope": "Common hardware and target manifest recovered from all hash-verified source chunk summaries; conditions retained per process.",
            "source_summary_files": metadata_sources,
            "verified_chunk_count": len(metadata_sources),
            "hardware_available": bool(common["hardware"]),
            "target_manifest_available": bool(common["model_manifest"]),
        }
        merged_hardware = cls._runtime(summary)["hardware"]
        merged_model = cls._model(summary)
        if any(common["hardware"].get(key) != value for key, value in merged_hardware.items()):
            raise ValueError("Merged hardware metadata disagrees with source chunks")
        if summary.get("model_manifest") and merged_model != common["model"]:
            raise ValueError("Merged target metadata disagrees with source chunks")
        return result, common

    @classmethod
    def _model(cls, summary):
        source = summary.get("model_manifest") or {}
        model = cls._pick(source, ("repo_id", "requested_revision", "resolved_revision", "format", "scope"))
        model["files"] = [{**cls._pick(item, ("sha256", "bytes")), "path": cls._path(item["path"])}
                          for item in source.get("files", []) if "path" in item]
        return model

    @classmethod
    def _runtime(cls, summary):
        backend, environment = summary.get("backend") or {}, summary.get("environment") or {}
        result = {"hardware": cls._pick(environment, ("os", "machine", "processor", "memory_bytes")),
                  "python": cls._text(environment.get("python")),
                  "packages": cls._versions(environment.get("packages")),
                  "backend": cls._pick(backend, cls.BACKEND_FIELDS),
                  "conditions_before": cls._conditions(environment),
                  "conditions_after": cls._conditions(summary.get("resources_after"))}
        result["backend"]["versions"] = cls._versions(backend.get("versions"))
        result["backend"]["device_info"] = cls._pick(backend.get("device_info") or {},
                ("device_name", "architecture", "memory_size", "max_recommended_working_set_size"))
        result["backend"]["upstream_runtime_sources"] = {
            cls._text(name): {"path": cls._path(item["path"]), "sha256": item["sha256"]}
            for name, item in (backend.get("upstream_runtime_sources") or {}).items()
            if "path" in item and "sha256" in item}
        return result

    @staticmethod
    def _eos_ids(value):
        if type(value) is int:
            value = [value]
        if isinstance(value, list) and value and all(type(token) is int and token >= 0 for token in value):
            return sorted(set(value))
        return None

    def _eos_config(self, summary):
        """Read EOS only from a config whose bytes match the recorded model manifest."""
        model_path = Path(summary.get("config", {}).get("model_path", ""))
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        files = {item["path"]: item for item in (summary.get("model_manifest") or {}).get("files", [])}
        for name in ("generation_config.json", "config.json"):
            if name not in files:
                continue
            path = model_path / name
            if not path.is_file():
                continue
            raw = path.read_bytes()
            if digest(raw) != files[name].get("sha256"):
                raise ValueError(f"EOS source hash mismatch: {name}")
            config = json.loads(raw)
            ids = self._eos_ids(config.get("eos_token_id")) or self._eos_ids(config.get("text_config", {}).get("eos_token_id"))
            if ids:
                return {"token_ids": ids, "source": self._path(path), "sha256": digest(raw)}
        return {"token_ids": None, "source": "unavailable"}

    def _eos(self, row, fallback):
        trace = row.get("generation_trace") or {}
        ids = self._eos_ids(trace.get("eos_token_ids")) or fallback["token_ids"]
        position = next((i for i, token in enumerate(row["generated_token_ids"]) if ids and token in ids), None)
        result = {"configured_token_ids": ids, "source": "generation_trace.eos_token_ids" if self._eos_ids(trace.get("eos_token_ids")) else fallback["source"],
                  "first_eos_index": position, "tokens_through_first_eos": position + 1 if position is not None else None,
                  "observed_within_budget": position is not None if ids is not None else None,
                  "time_to_first_eos_seconds": None, "time_source": "unavailable"}
        if position is not None and trace:
            for event in trace.get("events", []):
                if event.get("type") != "commit":
                    continue
                stop = event["output_count"]
                start = stop - len(event["token_ids"])
                if start <= position < stop:
                    result.update(time_to_first_eos_seconds=event["t"],
                                  time_source="commit.t, seconds since backend prefill start",
                                  commit_token_start_index=start, commit_token_end_index=stop - 1)
                    break
        return result

    def _sample(self, row, eos_config):
        total = row["prefill_seconds"] + row["decode_seconds"]
        result = {"index": row["index"], "prompt_token_sha256": row["prompt_token_sha256"],
                  "output_token_ids_sha256": SpeculativeReport._hash(row["generated_token_ids"]),
                  "raw_sample_object_sha256": object_digest(row),
                  **self._pick(row, ("prompt_tokens", "generated_tokens", "decode_tokens", "prefill_seconds",
                                     "decode_seconds", "peak_memory_gb", "timing_method", *self.COUNTERS)),
                  "prefill_tokens_per_second": row["prompt_tokens"] / row["prefill_seconds"],
                  "decode_tokens_per_second": row["decode_tokens"] / row["decode_seconds"],
                  "end_to_end_seconds": total, "end_to_end_output_tokens_per_second": row["generated_tokens"] / total,
                  "eos": self._eos(row, eos_config)}
        if "generation_trace" in row:
            result["generation_trace_sha256"] = object_digest(row["generation_trace"])
            result["instrumentation"] = self._pick(row["generation_trace"].get("instrumentation") or {},
                ("enabled", "version", "clock", "observer_overhead_included", "baseline", "speculative", "display_decoding", "comparability"))
        return result

    def _run(self, directory, source):
        # Re-read the validated bytes, closing the validation/export race.
        data = {}
        for name, item in source["files"].items():
            raw = Path(item["path"]).read_bytes()
            if digest(raw) != item["sha256"]:
                raise ValueError(f"Source changed after validation: {name}")
            data[name] = raw.decode()
        summary = json.loads(data["summary.json"])
        rows = [json.loads(line) for line in data["samples.jsonl"].splitlines() if line.strip()]
        config = summary["config"]
        prompts = json.loads(data["prompts.json"])
        series, common_metadata = self._series(summary, prompts, rows)
        metadata_summary = {**summary, "model_manifest": common_metadata["model_manifest"]} if common_metadata else summary
        runtime = self._runtime(summary)
        runtime["host_guard"] = series["host_guard"] if series else {
            "status": "not_guarded", "scope": "Standalone run has no suite sleep/power guard evidence; caffeinate does not prove absence of sleep."}
        if common_metadata:
            runtime["hardware"] = common_metadata["hardware"]
            runtime["hardware_scope"] = "Common values across every hash-verified source chunk; see series.metadata_provenance."
            runtime["conditions_before"] = runtime["conditions_after"] = None
            runtime["conditions_scope"] = "Per-process snapshots in series.source_block_runs; merged statistics have no single before/after snapshot."
        if config.get("trace_generation", False) and any(not isinstance(row.get("generation_trace"), dict) for row in rows):
            raise ValueError("Instrumented run is missing a generation trace")
        from inference_lab.visualization.recording import validate_generation_trace
        for row in rows:
            if "generation_trace" in row:
                validate_generation_trace(row)
                if not config.get("trace_generation", False):
                    raise ValueError("Generation trace contradicts uninstrumented protocol")
        eos_config = ({"token_ids": None, "source": "declared in every generation trace"}
                      if all(self._eos_ids((row.get("generation_trace") or {}).get("eos_token_ids")) for row in rows)
                      else self._eos_config(metadata_summary))
        samples = [self._sample(row, eos_config) for row in rows]
        stats = aggregate(rows)
        durations = [row["end_to_end_seconds"] for row in samples]
        rates = [row["end_to_end_output_tokens_per_second"] for row in samples]
        stats["end_to_end"] = {**distribution(rates), "unit": "output tokens/s", "total_seconds": sum(durations),
                              "total_output_tokens": sum(row["generated_tokens"] for row in rows),
                              "aggregate_tokens_per_second": sum(row["generated_tokens"] for row in rows) / sum(durations)}
        observed = [row["eos"]["observed_within_budget"] for row in samples]
        stats["eos"] = {"samples_with_known_eos": sum(value is not None for value in observed),
                        "samples_with_first_eos": sum(value is True for value in observed),
                        "samples_without_eos_within_budget": sum(value is False for value in observed),
                        "completed_within_budget": sum(value is True for value in observed) if all(value is not None for value in observed) else None,
                        "meaning": "Observed configured EOS within a fixed output budget; not an answer-correctness assessment. EOS did not stop generation."}
        eos_times = [row["eos"]["time_to_first_eos_seconds"] for row in samples
                     if row["eos"]["time_to_first_eos_seconds"] is not None]
        stats["eos"]["time_to_first_eos_seconds"] = distribution(eos_times) if eos_times else None
        protocol = self._pick(config, (*SpeculativeReport.PROTOCOL, *SpeculativeReport.OPTIONAL_PROTOCOL_DEFAULTS,
                                      "backend", "label", "wired_memory", "block_size", "draft_bits"))
        for key in ("model_path", "dataset_path", "draft_path"):
            if config.get(key):
                protocol[key] = self._path(config[key])
        return {"directory": self._path(directory), "sources": self._files(source["files"]),
                "started_at": self._text(summary.get("started_at")), "protocol": protocol,
                **{key: summary[key] for key in ("model_manifest_sha256", "dataset_sha256", "prompt_tokens_sha256")},
                "model": self._model(metadata_summary), "runtime": runtime, "eos_config": eos_config,
                "series": series, "statistics": stats, "samples": samples}

    @staticmethod
    def _paired_series(baseline, speculative):
        """Check the full paired schedule, not only each role's independent coverage."""
        if baseline.get("host_guard_policy") != speculative.get("host_guard_policy"):
            raise ValueError("Paired source groups declare different host guard policies")
        left = {item["chunk_index"]: item for item in baseline["source_block_runs"]}
        right = {item["chunk_index"]: item for item in speculative["source_block_runs"]}
        if set(left) != set(right) or sorted(left) != list(range(len(left))):
            raise ValueError("Paired source chunk indices differ or are not contiguous")
        expected_order = []
        for index in sorted(left):
            a, b = left[index], right[index]
            if (a["start_index"], a["count"]) != (b["start_index"], b["count"]):
                raise ValueError("Paired source chunk prompt ranges differ")
            expected_order.extend([a["id"], b["id"]] if index % 2 == 0 else [b["id"], a["id"]])
        if (len(set(expected_order)) != len(expected_order)
                or baseline["execution_order"] != expected_order
                or speculative["execution_order"] != expected_order):
            raise ValueError("Paired execution_order has missing/extra/duplicate IDs or conflicts with ABBA order")

    def analyze(self):
        validation = SpeculativeReport(self.baseline, [self.speculative]).analyze()
        report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                  "status": "failed", "parity_status": "unavailable", "unsuitable_for_lossless_claim": True,
                  "errors": [], "warnings": [self._text(item["message"]) for item in validation.get("warnings", [])]}
        if validation["status"] != "completed":
            report["errors"] = [self._text(item["message"]) for item in validation["errors"]]
            return report
        try:
            baseline = self._run(self.baseline, validation["sources"][str(self.baseline)])
            speculative = self._run(self.speculative, validation["sources"][str(self.speculative)])
            if bool(baseline["series"]) != bool(speculative["series"]):
                raise ValueError("Both runs must share the merged-series provenance scope")
            if baseline["series"] and speculative["series"]:
                self._paired_series(baseline["series"], speculative["series"])
                if (baseline["runtime"]["hardware"] != speculative["runtime"]["hardware"]
                        or baseline["model"] != speculative["model"]):
                    raise ValueError("Common hardware or target model differs between paired source chunk groups")
            if baseline["protocol"].get("trace_generation", False):
                common_policy = ("enabled", "version", "clock", "observer_overhead_included", "display_decoding", "comparability")
                policies = [{key: row["instrumentation"].get(key) for key in common_policy}
                            for run in (baseline, speculative) for row in run["samples"]]
                if any(policy != policies[0] for policy in policies[1:]):
                    raise ValueError("Generation recording policies differ between measured samples")
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["errors"] = [self._text(str(error))]
            return report
        paired = []
        for old, new in zip(baseline["samples"], speculative["samples"]):
            paired.append({"index": old["index"], **{phase: old[f"{phase}_seconds"] / new[f"{phase}_seconds"]
                           for phase in ("prefill", "decode", "end_to_end")}})
        paired_stats = {}
        for phase in ("prefill", "decode", "end_to_end"):
            values = [row[phase] for row in paired]
            paired_stats[phase] = {**distribution(values), "faster_prompts": sum(x > 1 for x in values),
                                   "equal_prompts": sum(x == 1 for x in values), "slower_prompts": sum(x < 1 for x in values)}
        comparison = validation["comparisons"][0]
        speedup = comparison["speedup"]
        left, right = baseline["statistics"]["end_to_end"], speculative["statistics"]["end_to_end"]
        speedup["end_to_end"] = {"ratio_of_mean_rates": right["mean"] / left["mean"],
                                 "ratio_of_aggregate_rates": right["aggregate_tokens_per_second"] / left["aggregate_tokens_per_second"]}
        instrumented = baseline["protocol"].get("trace_generation", False)
        report.update(status="completed", parity_status=comparison["parity"]["status"],
                      unsuitable_for_lossless_claim=comparison["parity"]["status"] != "passed",
                      runs={"baseline": baseline, "speculative": speculative}, parity=comparison["parity"],
                      acceptance=comparison["speculation"], speedup=speedup,
                      paired_speedup={"definition": "baseline phase seconds / speculative phase seconds for the same prompt",
                                      "statistics": paired_stats, "per_prompt": paired},
                      methodology={"instrumented": instrumented, "fixed_output_budget": True,
                          "host_guard": baseline["runtime"]["host_guard"],
                          "timing_overhead": "Draft proposal host reads and baseline per-token clocks are included in instrumented measurements." if instrumented else "No generation trace requested by protocol.",
                          "end_to_end": "Sum of prefill and decode phases; numerator is all generated output tokens. Excludes loading, warmup and between-request overhead.",
                          "series": "When series metadata is present, each source chunk uses a separate process/model load and its own warmup; preserve chronological execution_order.",
                          "trace_provenance": "Raw generation traces and content stay local in the hashed samples.jsonl files; object hashes use sorted compact JSON.",
                          "limitations": [SpeculativeReport.NOTE,
                              "Instrumented measurements are a separate experiment from earlier uninstrumented short runs; do not transfer their speedup claims.",
                              "Native MTP uses an optimized quantized projection/argmax path while baseline materializes logits; this is an implementation comparison, not an isolated algorithm ablation.",
                              "Matching measured token IDs demonstrates parity only for these prompts and this finite budget; it is not a general proof of losslessness.",
                              "EOS is ignored while benchmarking. First-EOS timing is a commit timestamp, not an interpolated time for tokens inside a verified block."]})
        return report

    @staticmethod
    def markdown(report):
        lines = ["# Paired inference benchmark", "", f"Status: **{report['status']}**; token parity: **{report['parity_status']}**.", ""]
        if report["status"] != "completed":
            return "\n".join(lines + ["Publication refused: " + "; ".join(report["errors"]), ""])
        if report["unsuitable_for_lossless_claim"]:
            lines += ["**Output IDs differ: unsuitable for a lossless-equivalence claim.**", ""]
        baseline = report["runs"]["baseline"]
        protocol = baseline["protocol"]
        hardware, model = baseline["runtime"]["hardware"], baseline["model"]
        if hardware:
            lines += [f"Hardware: {hardware.get('processor', 'unavailable')}; memory {hardware.get('memory_bytes', 'unavailable')} bytes; {hardware.get('os', 'unavailable')}.", ""]
        if model.get("repo_id"):
            lines += [f"Target: `{model['repo_id']}` at `{model.get('resolved_revision', 'unavailable')}`.", ""]
        if baseline["series"]:
            lines += ["Hardware and target pins were checked across all source chunks. Per-chunk start time, runner elapsed time, power and thermal snapshots are preserved in JSON with source-summary SHA256 values.", ""]
        guard = report["methodology"]["host_guard"]
        lines += [f"Sleep/power guard: **{guard['status']}**. {guard['scope']}", ""]
        if guard.get("power_observation_limitation"):
            lines += [guard["power_observation_limitation"], ""]
        lines += [f"{protocol['count']} identical prompts × {protocol['max_new_tokens']} output tokens; instrumented: **{report['methodology']['instrumented']}**.",
                  report["methodology"]["timing_overhead"], "",
                  "Mean ± sample SD (ddof=1), tokens/s. Pooled = total tokens / total phase seconds.", "",
                  "| Phase | Baseline mean ± SD | Speculative mean ± SD | Pooled baseline / speculative | Mean-rate speedup |",
                  "|---|---:|---:|---:|---:|"]
        def fmt(value):
            return "n/a" if value is None else f"{value:.2f}"
        for phase in ("prefill", "decode", "end_to_end"):
            old, new = [report["runs"][role]["statistics"][phase] for role in ("baseline", "speculative")]
            mean_key, sd_key = ("mean", "sample_stddev") if phase == "end_to_end" else ("mean_tokens_per_second", "sample_stddev_tokens_per_second")
            lines.append(f"| {phase} | {fmt(old[mean_key])} ± {fmt(old[sd_key])} | {fmt(new[mean_key])} ± {fmt(new[sd_key])} | {fmt(old['aggregate_tokens_per_second'])} / {fmt(new['aggregate_tokens_per_second'])} | {report['speedup'][phase]['ratio_of_mean_rates']:.3f}× |")
        acceptance = report["acceptance"]
        lines += ["", f"Acceptance: {acceptance['accepted_draft_tokens']}/{acceptance['drafted_tokens']}; committed decode tokens per verification round: {acceptance['mean_committed_tokens_per_round']:.3f}.",
                  f"Exact parity: {report['parity']['matched_prompts']}/{report['parity']['total_prompts']} prompts.", ""]
        for role, run in report["runs"].items():
            eos = run["statistics"]["eos"]
            lines.append(f"- {role}: first EOS observed in {eos['samples_with_first_eos']}/{len(run['samples'])} prompts ({eos['samples_with_known_eos']} have known EOS configuration); source `{run['directory']}`.")
        lines += ["", report["methodology"]["end_to_end"], "", *["- " + text for text in report["methodology"]["limitations"]], "",
                  "The accompanying JSON contains per-prompt timings, sample variance, paired speedup distributions, source hashes, runtime pins and chunk order; original raw files are unchanged.", ""]
        return "\n".join(lines)

    def build(self):
        report = self.analyze()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        self.output.with_suffix(".md").write_text(self.markdown(report))
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--speculative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Public output basename, e.g. docs/results/mtp-100-2048")
    args = parser.parse_args(argv)
    report = PublicationExporter(args.baseline, args.speculative, args.output).build()
    print(json.dumps({key: report[key] for key in ("status", "parity_status", "errors")}))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
