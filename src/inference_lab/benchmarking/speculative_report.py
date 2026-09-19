"""Compare explicitly selected speculative runs with a matching target baseline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from inference_lab.core.config import ROOT
from inference_lab.core.io import write_json
from .metrics import aggregate, validate_measurement
from .recheck_report import RecheckReport


class SpeculativeReport:
    """Fail closed on incomparable inputs; retain and disclose output mismatches."""

    PROTOCOL = ("count", "max_new_tokens", "max_prompt_tokens", "prefill_step_size",
                "warmup", "prompt_mode", "chain_prefix_tokens", "kv_bits", "user_initiated")
    MLX_PACKAGES = ("mlx", "mlx-lm", "mlx-metal")
    BACKEND_FAMILIES = {"mlx": ("mlx-dflash", "spec-baseline"),
                        "mlx-vlm": ("mlx-mtp", "mtp-baseline")}
    NOTE = ("Среднее ± выборочное SD (ddof=1) рассчитаны заново из samples.jsonl. "
            "Для одного запроса SD и variance не определены. Разброс между разными prompts "
            "не является доверительным интервалом и не оценивает дисперсию повторных запусков одного prompt. "
            "Prefill включает первый выходной токен; decode учитывает только оставшиеся полезные токены.")

    def __init__(self, baseline: Path, speculative: list[Path],
                 output: Path = ROOT / "artifacts/reports"):
        self.baseline = Path(baseline).resolve()
        self.speculative = [Path(path).resolve() for path in speculative]
        self.output = Path(output)

    @staticmethod
    def _hash(value):
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()

    @classmethod
    def _package_names(cls, backend):
        return cls.MLX_PACKAGES + (("mlx-vlm",) if backend in ("mlx-vlm", "mlx-mtp") else ())

    @classmethod
    def _versions(cls, summary):
        versions = {}
        package_names = cls._package_names(summary.get("config", {}).get("backend"))
        groups = [summary.get("environment", {}).get("packages", {}),
                  summary.get("backend", {}).get("versions", {}),
                  summary.get("backend", {}).get("packages", {})]
        for group in groups:
            for name, version in (group or {}).items():
                name = name.lower().replace("_", "-")
                if name not in package_names or version is None:
                    continue
                if name in versions and versions[name] != version:
                    raise ValueError(f"Conflicting environment/backend version for {name}")
                versions[name] = version
        return versions

    @staticmethod
    def _counters(row):
        """Both speculative adapters store counters directly in each sample."""
        keys = ("drafted_tokens", "accepted_draft_tokens", "speculative_rounds")
        for key in keys:
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"Invalid speculative counter: {key}")
        if row["speculative_rounds"] < 1:
            raise ValueError("Speculative sample must have at least one verification round")
        if row["accepted_draft_tokens"] > row["drafted_tokens"]:
            raise ValueError("Accepted draft tokens exceed drafted tokens")
        for key in ("emitted_draft_tokens", "emitted_target_tokens"):
            if key in row and (type(row[key]) is not int or row[key] < 0):
                raise ValueError(f"Invalid speculative counter: {key}")
        if "emitted_draft_tokens" in row and "emitted_target_tokens" in row:
            if row["emitted_draft_tokens"] + row["emitted_target_tokens"] != row["generated_tokens"]:
                raise ValueError("Emitted draft and target tokens do not match generated tokens")
            if row["emitted_draft_tokens"] > row["accepted_draft_tokens"]:
                raise ValueError("Emitted draft tokens exceed accepted draft tokens")
        return {key: row[key] for key in keys}

    def _load(self, directory, role, evidence):
        # Preserve hashes of every readable source even when validation fails.
        files, data = evidence.setdefault("files", {}), {}
        for name in ("summary.json", "prompts.json", "samples.jsonl"):
            path = directory / name
            raw = path.read_bytes()
            files[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
            data[name] = raw.decode()
        summary = json.loads(data["summary.json"])
        prompts = json.loads(data["prompts.json"])
        rows = [json.loads(line) for line in data["samples.jsonl"].splitlines() if line.strip()]
        if not isinstance(summary, dict):
            raise ValueError("summary.json must contain an object")
        config = summary.get("config", {})
        count = config.get("count")
        maximum = config.get("max_new_tokens")
        if type(count) is not int or count < 1 or type(maximum) is not int or maximum < 2:
            raise ValueError("Protocol requires count >= 1 and max_new_tokens >= 2")
        if summary.get("status") != "completed" or summary.get("completed_samples") != count:
            raise ValueError("Run must be completed with every configured sample")
        allowed_backends = (self.BACKEND_FAMILIES if role == "baseline" else
                            {family[0] for family in self.BACKEND_FAMILIES.values()})
        if config.get("backend") not in allowed_backends:
            raise ValueError(f"Unsupported {role} backend: {config.get('backend')}")
        if role == "baseline":
            label = self.BACKEND_FAMILIES[config["backend"]][1]
            if not re.fullmatch(re.escape(label) + r"(?:-[A-Za-z0-9_-]+)?", config.get("label", "")):
                raise ValueError(f"Baseline label must belong to the {label} family")
        if summary.get("environment", {}).get("caffeinate_assertions") != "di":
            raise ValueError("Expected caffeinate_assertions=di")
        if config.get("wired_memory") is not True:
            raise ValueError("Expected wired_memory=true")
        backend = summary.get("backend") or {}
        if "wired_memory" in backend and backend["wired_memory"] is not True:
            raise ValueError("Backend metadata contradicts wired_memory=true")
        if "sampling" in backend and backend["sampling"] != "greedy":
            raise ValueError("Expected greedy sampling")
        for key, expected in (("ignore_eos", True), ("batch_size", 1),
                              ("fresh_cache_per_request", True), ("first_generated_token_phase", "prefill")):
            if key in backend and backend[key] != expected:
                raise ValueError(f"Unsupported backend policy: {key}")
        for key in self.PROTOCOL:
            if key not in config:
                raise ValueError(f"Missing protocol field: {key}")
        for key in ("model_manifest_sha256", "dataset_sha256"):
            if not isinstance(summary.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", summary[key]):
                raise ValueError(f"Missing or invalid {key}")
        if not isinstance(prompts, list) or len(prompts) != count or len(rows) != count:
            raise ValueError("Raw prompt/sample count differs from the completed protocol")
        indices = []
        for position, (prompt, row) in enumerate(zip(prompts, rows)):
            if not isinstance(prompt, dict) or not isinstance(row, dict):
                raise ValueError(f"Invalid prompt/sample at position {position}")
            index = prompt.get("index")
            if type(index) is not int or index < 0 or row.get("index") != index:
                raise ValueError(f"Prompt/sample index mismatch at position {position}")
            indices.append(index)
            tokens = prompt.get("prompt_tokens")
            if (not isinstance(tokens, list) or not tokens
                    or any(type(token) is not int or token < 0 for token in tokens)):
                raise ValueError(f"Invalid input token IDs at position {position}")
            if row.get("prompt_token_sha256") != self._hash(tokens):
                raise ValueError(f"Input token hash mismatch at position {position}")
            validate_measurement(row)
            if row["prompt_tokens"] != len(tokens) or row["generated_tokens"] != maximum:
                raise ValueError(f"Token count differs from protocol at position {position}")
            generated = row.get("generated_token_ids")
            if (not isinstance(generated, list) or len(generated) != maximum
                    or any(type(token) is not int or token < 0 for token in generated)):
                raise ValueError(f"Invalid generated token IDs at position {position}")
            if not isinstance(row.get("timing_method"), str) or not row["timing_method"]:
                raise ValueError(f"Missing timing method at position {position}")
            if role == "speculative":
                self._counters(row)
        if len(set(indices)) != count:
            raise ValueError("Duplicate prompt indices")
        if summary.get("prompt_tokens_sha256") != self._hash([prompt["prompt_tokens"] for prompt in prompts]):
            raise ValueError("Combined input token hash differs from summary")
        source = {"directory": str(directory), "files": files, "config": config,
                  "started_at": summary.get("started_at"), "backend": backend,
                  "environment": summary.get("environment"), "resources_after": summary.get("resources_after"),
                  "mlx_versions": self._versions(summary),
                  "timing_methods": sorted({row["timing_method"] for row in rows}),
                  **{key: summary[key] for key in ("model_manifest_sha256", "dataset_sha256", "prompt_tokens_sha256")}}
        return {"source": source, "prompts": prompts, "rows": rows, "summary": summary}

    @staticmethod
    def _parity(baseline, speculative):
        records = []
        for old, new in zip(baseline, speculative):
            left, right = old["generated_token_ids"], new["generated_token_ids"]
            prefix = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            same = left == right
            records.append({"index": old["index"], "matched": same, "matching_prefix_tokens": prefix,
                            "first_mismatch_token_position": None if same else prefix,
                            "baseline_token_id": None if same or prefix >= len(left) else left[prefix],
                            "speculative_token_id": None if same or prefix >= len(right) else right[prefix]})
        return {"status": "passed" if all(row["matched"] for row in records) else "failed",
                "matched_prompts": sum(row["matched"] for row in records), "total_prompts": len(records),
                "token_position_convention": "zero-based", "per_prompt": records}

    def analyze(self):
        report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                  "status": "failed", "parity_status": "unavailable", "note": self.NOTE,
                  "errors": [], "warnings": [], "sources": {}, "comparisons": []}
        selected = [("baseline", self.baseline)] + [("speculative", path) for path in self.speculative]
        if not self.speculative or len({path for _, path in selected}) != len(selected):
            report["errors"].append({"message": "Select at least one distinct speculative run and a separate baseline"})
            return report
        loaded = []
        for role, directory in selected:
            report["sources"][str(directory)] = {"directory": str(directory), "role": role}
            try:
                run = self._load(directory, role, report["sources"][str(directory)])
                loaded.append(run)
                report["sources"][str(directory)] = {"role": role, **run["source"]}
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                report["errors"].append({"directory": str(directory), "message": str(error)})
        if report["errors"]:
            return report
        baseline, *candidates = loaded
        baseline_backend = baseline["summary"]["config"]["backend"]
        expected_speculative = self.BACKEND_FAMILIES[baseline_backend][0]
        for run in candidates:
            differences = []
            if run["summary"]["config"]["backend"] != expected_speculative:
                differences.append(f"backend family: {baseline_backend} requires {expected_speculative}")
            for key in ("model_manifest_sha256", "dataset_sha256", "prompt_tokens_sha256"):
                if run["source"][key] != baseline["source"][key]:
                    differences.append(key)
            for key in self.PROTOCOL:
                if run["summary"]["config"][key] != baseline["summary"]["config"][key]:
                    differences.append(f"config.{key}")
            left = [(p["index"], p["prompt_tokens"]) for p in baseline["prompts"]]
            right = [(p["index"], p["prompt_tokens"]) for p in run["prompts"]]
            if left != right:
                differences.append("exact input prompt indices/token IDs")
            for name in self._package_names(baseline_backend):
                a, b = baseline["source"]["mlx_versions"].get(name), run["source"]["mlx_versions"].get(name)
                if a is not None and b is not None and a != b:
                    differences.append(f"package {name}: {a} != {b}")
                elif a is None or b is None:
                    report["warnings"].append({"directory": run["source"]["directory"],
                                               "message": f"Version equivalence for {name} unavailable in one or both sources"})
            if differences:
                report["errors"].append({"directory": run["source"]["directory"],
                                         "message": "Incompatible inputs: " + ", ".join(differences)})
        if report["errors"]:
            return report
        report["baseline"] = {"directory": str(self.baseline), "statistics": aggregate(baseline["rows"])}
        for run in candidates:
            rows = run["rows"]
            statistics = aggregate(rows)
            drafted = sum(row["drafted_tokens"] for row in rows)
            accepted = sum(row["accepted_draft_tokens"] for row in rows)
            rounds = sum(row["speculative_rounds"] for row in rows)
            committed = sum(row["decode_tokens"] for row in rows)
            comparison = {
                "directory": run["source"]["directory"], "label": run["source"]["config"].get("label"),
                "statistics": statistics, "parity": self._parity(baseline["rows"], rows),
                "speedup": {phase: {"ratio_of_mean_rates": statistics[phase]["mean_tokens_per_second"] /
                                    report["baseline"]["statistics"][phase]["mean_tokens_per_second"],
                                    "ratio_of_aggregate_rates": statistics[phase]["aggregate_tokens_per_second"] /
                                    report["baseline"]["statistics"][phase]["aggregate_tokens_per_second"]}
                            for phase in ("prefill", "decode")},
                "speculation": {"drafted_tokens": drafted, "accepted_draft_tokens": accepted,
                                "weighted_draft_acceptance_rate": accepted / drafted if drafted else None,
                                "speculative_rounds": rounds, "committed_decode_tokens": committed,
                                "mean_committed_tokens_per_round": committed / rounds,
                                "mean_accepted_draft_tokens_per_round": accepted / rounds,
                                "accepted_tokens_scope": "all verified matching draft tokens; final length cap may omit accepted tokens from output",
                                "committed_tokens_scope": "decode only; first prefill token excluded"},
            }
            report["comparisons"].append(comparison)
        report["status"] = "completed"
        report["parity_status"] = "passed" if all(row["parity"]["status"] == "passed" for row in report["comparisons"]) else "failed"
        return report

    @staticmethod
    def _rate(stats):
        deviation = stats["sample_stddev_tokens_per_second"]
        return f"{stats['mean_tokens_per_second']:.2f} ± {deviation:.2f}" if deviation is not None else f"{stats['mean_tokens_per_second']:.2f} (SD н/д)"

    def _markdown(self, report):
        lines = ["# Проверка speculative decoding", "", self.NOTE, ""]
        if report["status"] != "completed":
            lines += ["**Сравнение не выполнено: входные данные неполны или несовместимы.**", ""]
            lines.extend(f"- {error.get('directory', '')}: {error['message']}" for error in report["errors"])
        else:
            if report["parity_status"] == "failed":
                lines += ["**Проверка выходных token IDs не пройдена. Скорости относятся к генерациям с различающимися "
                          "результатами; точная эквивалентность baseline не подтверждена.**", ""]
            else:
                lines += ["Выходные token IDs совпали во всех проверенных prompts. Это наблюдение ограничено этой выборкой "
                          "и не доказывает эквивалентность на произвольных входах.", ""]
            baseline = report["baseline"]["statistics"]
            lines += ["| Прогон | N | Prefill mean ± SD, tok/s | Decode mean ± SD, tok/s | Decode × mean | Decode × aggregate | Parity |",
                      "|---|---:|---:|---:|---:|---:|---|",
                      f"| baseline | {baseline['samples']} | {self._rate(baseline['prefill'])} | {self._rate(baseline['decode'])} | 1.000 | 1.000 | reference |"]
            mismatches = []
            for row in report["comparisons"]:
                stats, speed = row["statistics"], row["speedup"]["decode"]
                lines.append(f"| {row['label']} | {stats['samples']} | {self._rate(stats['prefill'])} | {self._rate(stats['decode'])} | "
                             f"{speed['ratio_of_mean_rates']:.3f} | {speed['ratio_of_aggregate_rates']:.3f} | {row['parity']['status']} |")
            lines += ["", "Отношение mean сравнивает арифметические средние скоростей запросов. "
                      "Aggregate = сумма полезных токенов / сумма времени; это отдельная оценка.", "",
                      "| Прогон | Draft принято / предложено | Acceptance | Verification rounds | Полезных decode-токенов / round |",
                      "|---|---:|---:|---:|---:|"]
            for row in report["comparisons"]:
                spec = row["speculation"]
                acceptance = spec["weighted_draft_acceptance_rate"]
                percent = f"{100 * acceptance:.2f}%" if acceptance is not None else "н/д"
                lines.append(f"| {row['label']} | {spec['accepted_draft_tokens']} / {spec['drafted_tokens']} | {percent} | "
                             f"{spec['speculative_rounds']} | {spec['mean_committed_tokens_per_round']:.3f} |")
                for parity in row["parity"]["per_prompt"]:
                    if not parity["matched"]:
                        mismatches.append(f"- {row['label']}, prompt {parity['index']}: первое несовпадение на позиции "
                                          f"{parity['first_mismatch_token_position']} (нумерация с 0), общий префикс "
                                          f"{parity['matching_prefix_tokens']} токенов; baseline ID {parity['baseline_token_id']}, "
                                          f"speculative ID {parity['speculative_token_id']}.")
            lines += ["", "Acceptance = сумма принятых draft-токенов / сумма предложенных. "
                      "Принятые при проверке токены могут не попасть в вывод из-за ограничения его длины в последнем блоке. "
                      "Полезные токены / round учитывают фактически выведенные decode-токены; первый токен prefill исключён."]
            if mismatches:
                lines += ["", *mismatches]
            lines += ["", "Проверенные ограничения: одинаковые входные token IDs, target/dataset hashes и протокол; "
                      "caffeinate `di`, wired memory. Доступные версии MLX сверены. Питание и температура "
                      "этими ограничениями не фиксируются; причинное влияние их изменений не измерено.", ""]
            for directory, source in report["sources"].items():
                lines.append(f"- {Path(directory).name}: питание в начале → конце: {RecheckReport._power(source)}.")
        if report["warnings"]:
            lines += ["", "Ограничения метаданных:", ""]
            lines.extend(f"- {warning['message']} ({Path(warning['directory']).name})" for warning in report["warnings"])
        lines += ["", "Исходные файлы (SHA-256 каждого файла записан в JSON/CSV):", ""]
        for directory, source in report["sources"].items():
            lines.append(f"- {source['role']}: [{Path(directory).name}]({directory}/summary.json)")
        lines += ["", "Variance s² доступна в JSON/CSV в единицах (tok/s)². Исходные файлы не изменены.", ""]
        return "\n".join(lines)

    def build(self):
        report = self.analyze()
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "speculative-comparison.json", report)
        (self.output / "speculative-comparison.md").write_text(self._markdown(report))
        fields = ["status", "role", "directory", "label", "phase", "samples", "mean_tokens_per_second",
                  "sample_stddev_tokens_per_second", "sample_variance_tokens_per_second", "aggregate_tokens_per_second",
                  "speedup_ratio_of_means", "speedup_ratio_of_aggregates", "parity_status", "matched_prompts",
                  "drafted_tokens", "accepted_draft_tokens", "weighted_draft_acceptance_rate", "speculative_rounds",
                  "mean_committed_tokens_per_round", "model_manifest_sha256", "dataset_sha256",
                  "summary_sha256", "prompts_sha256", "samples_sha256", "baseline_directory",
                  "baseline_summary_sha256", "baseline_prompts_sha256", "baseline_samples_sha256", "error"]
        with (self.output / "speculative-comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            if report["status"] == "completed":
                records = [{**report["baseline"], "role": "baseline"}] + [{**item, "role": "speculative"} for item in report["comparisons"]]
                baseline_source = report["sources"][str(self.baseline)]
                for item in records:
                    source = report["sources"][item["directory"]]
                    for phase in ("prefill", "decode"):
                        stats = item["statistics"][phase]
                        row = {"status": "completed", "role": item["role"], "directory": item["directory"],
                               "label": source["config"]["label"], "phase": phase, "samples": item["statistics"]["samples"],
                               **{key: stats[key] for key in fields[6:10]},
                               "speedup_ratio_of_means": item.get("speedup", {}).get(phase, {}).get("ratio_of_mean_rates", 1),
                               "speedup_ratio_of_aggregates": item.get("speedup", {}).get(phase, {}).get("ratio_of_aggregate_rates", 1),
                               "parity_status": item.get("parity", {}).get("status", "reference"),
                               "matched_prompts": item.get("parity", {}).get("matched_prompts", ""),
                               **{key: item.get("speculation", {}).get(key, "") for key in fields[14:19]},
                               "model_manifest_sha256": source["model_manifest_sha256"], "dataset_sha256": source["dataset_sha256"],
                               "baseline_directory": str(self.baseline)}
                        for prefix, filename in (("summary", "summary.json"), ("prompts", "prompts.json"), ("samples", "samples.jsonl")):
                            row[f"{prefix}_sha256"] = source["files"][filename]["sha256"]
                            row[f"baseline_{prefix}_sha256"] = baseline_source["files"][filename]["sha256"]
                        writer.writerow(row)
            for error in report["errors"]:
                writer.writerow({"status": "failed", "directory": error.get("directory", ""), "error": error["message"]})
        print(self.output / "speculative-comparison.md")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--speculative", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reports")
    args = parser.parse_args(argv)
    report = SpeculativeReport(args.baseline, args.speculative, args.output).build()
    if report["status"] != "completed":
        raise SystemExit(1)
