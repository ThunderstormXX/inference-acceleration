"""Compare an exact five-prompt repeat with immutable awake-run evidence."""

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


class RecheckReport:
    BACKENDS = ("transformers", "mlx", "mlx-vlm", "vllm")
    COHORTS = {"awake100": ("awake", 100), "recheck5": ("recheck5", 5)}
    PROTOCOL = ("max_new_tokens", "max_prompt_tokens", "prefill_step_size", "warmup",
                "prompt_mode", "chain_prefix_tokens", "kv_bits", "user_initiated")
    NOTE = ("Среднее и выборочное SD (ddof=1) рассчитаны по скоростям отдельных запросов. "
            "Разброс между пятью разными prompt не является доверительным интервалом "
            "и не оценивает дисперсию повторных запусков одного prompt. "
            "Delta сравнивает повтор5 именно с первыми5 старого прогона; старые100 показаны отдельно.")

    def __init__(self, runs: Path = ROOT / "artifacts/runs",
                 output: Path = ROOT / "artifacts/reports"):
        self.runs, self.output = Path(runs), Path(output)

    @staticmethod
    def _token_hash(tokens):
        return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()

    @staticmethod
    def _complete(summary, count):
        return (summary.get("status") == "completed"
                and summary.get("completed_samples") == count
                and summary.get("config", {}).get("count") == count)

    def _discover(self):
        candidates, unreadable = [], []
        for path in sorted(self.runs.glob("*/summary.json")):
            try:
                summary = json.loads(path.read_text())
                if not isinstance(summary, dict):
                    raise ValueError("summary must be an object")
                candidates.append((path, summary))
            except (OSError, ValueError) as exc:
                unreadable.append({"path": str(path.resolve()), "reason": str(exc)})
        return candidates, unreadable

    def _select(self, candidates, backend, cohort):
        label, count = self.COHORTS[cohort]
        matches, rejected = [], []
        for path, summary in candidates:
            config = summary.get("config", {})
            if (config.get("backend"), config.get("label")) != (backend, label):
                continue
            reason = None
            if config.get("count") != count:
                reason = f"expected count={count}"
            elif summary.get("environment", {}).get("caffeinate_assertions") != "di":
                reason = "expected caffeinate_assertions=di"
            elif not self._complete(summary, count):
                reason = "run is not completed with the requested sample count"
            if reason:
                rejected.append({"backend": backend, "cohort": cohort, "path": str(path.resolve()),
                                 "status": summary.get("status"), "reason": reason})
            else:
                matches.append((path, summary))
        if not matches:
            return None, rejected
        selected = max(matches, key=lambda item: (item[1].get("started_at", ""), item[0].parent.name))
        for path, _ in matches:
            if path != selected[0]:
                rejected.append({"backend": backend, "cohort": cohort, "path": str(path.resolve()),
                                 "reason": "older completed matching run"})
        return selected[0], rejected

    def _load(self, path, backend, cohort):
        files, contents = {}, {}
        for name in ("summary.json", "prompts.json", "samples.jsonl"):
            source = path.parent / name
            raw = source.read_bytes()
            contents[name] = raw.decode()
            files[name] = {"path": str(source.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
        summary = json.loads(contents["summary.json"])
        prompts = json.loads(contents["prompts.json"])
        rows = [json.loads(line) for line in contents["samples.jsonl"].splitlines() if line.strip()]
        label, count = self.COHORTS[cohort]
        config = summary["config"]
        if (not self._complete(summary, count) or config["backend"] != backend
                or config["label"] != label
                or summary.get("environment", {}).get("caffeinate_assertions") != "di"):
            raise ValueError("Selected summary no longer matches requested completed cohort")
        if not isinstance(prompts, list) or len(prompts) != count or len(rows) != count:
            raise ValueError(f"Raw source count mismatch: expected {count} prompts and samples")
        for key in (*self.PROTOCOL, "wired_memory"):
            if key not in config:
                raise ValueError(f"Missing protocol field: {key}")
        for key in ("dataset_sha256", "model_manifest_sha256"):
            if not isinstance(summary.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", summary[key]):
                raise ValueError(f"Missing or invalid {key}")
        for index, (prompt, row) in enumerate(zip(prompts, rows)):
            if not isinstance(prompt, dict) or not isinstance(row, dict):
                raise ValueError(f"Invalid raw record at row {index}")
            tokens = prompt.get("prompt_tokens")
            if (not isinstance(tokens, list) or not tokens
                    or any(type(token) is not int or token < 0 for token in tokens)):
                raise ValueError(f"Invalid input token IDs at row {index}")
            if prompt.get("index") != index or row.get("index") != index:
                raise ValueError(f"Raw prompt/sample index mismatch at row {index}")
            if row.get("prompt_token_sha256") != self._token_hash(tokens):
                raise ValueError(f"Prompt token hash mismatch at row {index}")
            validate_measurement(row)
            if row["prompt_tokens"] != len(tokens) or row["generated_tokens"] != config["max_new_tokens"]:
                raise ValueError(f"Token count differs from protocol at row {index}")
            generated = row.get("generated_token_ids")
            if (not isinstance(generated, list) or len(generated) != row["generated_tokens"]
                    or any(type(token) is not int or token < 0 for token in generated)):
                raise ValueError(f"Invalid generated token IDs at row {index}")
            if not isinstance(row.get("timing_method"), str) or not row["timing_method"]:
                raise ValueError(f"Missing timing method at row {index}")
        if summary.get("prompt_tokens_sha256") != self._token_hash([p["prompt_tokens"] for p in prompts]):
            raise ValueError("Combined prompt token hash does not match summary")
        evidence = {
            "files": files, "started_at": summary.get("started_at"), "config": config,
            "completed_samples": count, "dataset_sha256": summary["dataset_sha256"],
            "model_manifest_sha256": summary["model_manifest_sha256"],
            "model_revision": summary.get("model_manifest", {}).get("resolved_revision"),
            "prompt_tokens_sha256": summary["prompt_tokens_sha256"],
            "first5_prompt_tokens_sha256": self._token_hash([p["prompt_tokens"] for p in prompts[:5]]),
            "first5_indices": [p["index"] for p in prompts[:5]],
            "timing_methods": sorted({row["timing_method"] for row in rows}),
            "environment": {key: summary.get("environment", {}).get(key) for key in
                            ("os", "machine", "processor", "memory_bytes", "packages",
                             "caffeinate_assertions", "power_source", "thermal_state")},
            "resources_after": summary.get("resources_after"),
            "backend_metadata": summary.get("backend"),
        }
        return {"source": evidence, "summary": summary, "prompts": prompts, "rows": rows}

    def analyze(self):
        candidates, unreadable = self._discover()
        report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                  "status": "failed", "note": self.NOTE, "errors": [], "excluded_runs": [],
                  "unreadable_summaries": unreadable, "sources": {}, "comparisons": []}
        loaded = {}
        for backend in self.BACKENDS:
            report["sources"][backend] = {}
            for cohort in self.COHORTS:
                path, rejected = self._select(candidates, backend, cohort)
                report["excluded_runs"].extend(rejected)
                if path is None:
                    report["errors"].append({"backend": backend, "cohort": cohort,
                                             "message": "No completed matching run; partial/missing runs are not success"})
                    continue
                report["sources"][backend][cohort] = {"selected_summary": str(path.resolve())}
                try:
                    loaded[backend, cohort] = self._load(path, backend, cohort)
                    report["sources"][backend][cohort] = loaded[backend, cohort]["source"]
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    report["errors"].append({"backend": backend, "cohort": cohort, "message": str(exc)})
        if report["errors"]:
            return report

        reference = loaded[self.BACKENDS[0], "awake100"]
        for (backend, cohort), run in loaded.items():
            differences = []
            for key in ("dataset_sha256", "model_manifest_sha256"):
                if run["summary"][key] != reference["summary"][key]:
                    differences.append(key)
            for key in self.PROTOCOL:
                if run["summary"]["config"][key] != reference["summary"]["config"][key]:
                    differences.append(f"config.{key}")
            # The intended cross-backend memory policies differ; repeated runs
            # of the same backend must retain their own policy and phase boundary.
            baseline = loaded[backend, "awake100"]
            if run["summary"]["config"]["wired_memory"] != baseline["summary"]["config"]["wired_memory"]:
                differences.append("config.wired_memory within backend")
            if run["source"]["timing_methods"] != baseline["source"]["timing_methods"]:
                differences.append("timing_method within backend")
            for i, (prompt, expected) in enumerate(zip(run["prompts"][:5], reference["prompts"][:5])):
                if prompt["index"] != expected["index"] or prompt["prompt_tokens"] != expected["prompt_tokens"]:
                    differences.append(f"first5 prompt token IDs/index at position {i}")
            if cohort == "awake100" and run["summary"]["prompt_tokens_sha256"] != reference["summary"]["prompt_tokens_sha256"]:
                differences.append("awake100 full prompt token hash")
            if differences:
                report["errors"].append({"backend": backend, "cohort": cohort,
                                         "message": "Incompatible evidence: " + ", ".join(differences)})
        if report["errors"]:
            return report
        for backend in self.BACKENDS:
            old = loaded[backend, "awake100"]["rows"]
            new = loaded[backend, "recheck5"]["rows"]
            old5, new5 = aggregate(old[:5]), aggregate(new)
            report["comparisons"].append({
                "backend": backend, "old_first5": old5, "recheck5": new5,
                "awake100": aggregate(old),
                "delta_percent": {phase: 100 * (new5[phase]["mean_tokens_per_second"] /
                                                old5[phase]["mean_tokens_per_second"] - 1)
                                  for phase in ("prefill", "decode")},
            })
        report["status"] = "completed"
        return report

    @staticmethod
    def _format(stats, phase):
        return (f"{stats[phase]['mean_tokens_per_second']:.2f} ± "
                f"{stats[phase]['sample_stddev_tokens_per_second']:.2f}")

    @staticmethod
    def _power(source):
        """Display measured start/end power snapshots without asserting causality."""
        values = []
        for snapshot in (source.get("environment"), source.get("resources_after")):
            raw = ((snapshot or {}).get("power_source") or {}).get("raw") or ""
            power = re.search(r"Now drawing from '([^']+)'", raw)
            battery = re.search(r"(\d+)%;\s*([^;\n]+)", raw)
            text = power.group(1) if power else "н/д"
            if battery:
                text += f", {battery.group(1)}%, {battery.group(2).strip()}"
            values.append(text)
        return " → ".join(values)

    def _markdown(self, report):
        lines = ["# Повторная проверка на пяти одинаковых примерах", "", self.NOTE, "",
                 "Variance s² хранится в JSON/CSV в единицах (tok/s)²; SD — в tok/s. "
                 "Статистики заново рассчитаны из samples.jsonl; исходные summaries не изменены.", ""]
        if report["status"] != "completed":
            lines += ["**Проверка не завершена: сравнение скоростей не опубликовано.**", ""]
            lines += [f"- {error['backend']} / {error['cohort']}: {error['message']}" for error in report["errors"]]
        else:
            lines += ["| Backend | Prefill старые5 | Prefill повтор5 | Δ % | Decode старые5 | Decode повтор5 | Δ % |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for row in report["comparisons"]:
                lines.append(f"| {row['backend']} | {self._format(row['old_first5'], 'prefill')} | "
                             f"{self._format(row['recheck5'], 'prefill')} | {row['delta_percent']['prefill']:+.2f} | "
                             f"{self._format(row['old_first5'], 'decode')} | {self._format(row['recheck5'], 'decode')} | "
                             f"{row['delta_percent']['decode']:+.2f} |")
            lines += ["", "Все скорости в tok/s; значения в таблице — mean ± sample SD.", "",
                      "## Старые полные прогоны: 100 примеров awake", "",
                      "| Backend | Prefill mean ± SD, tok/s | Decode mean ± SD, tok/s |",
                      "|---|---:|---:|"]
            lines += [f"| {row['backend']} | {self._format(row['awake100'], 'prefill')} | "
                      f"{self._format(row['awake100'], 'decode')} |" for row in report["comparisons"]]
            lines += ["", "Проверены совпадение модели и датасета по SHA256, протокол нагрузки и точные token IDs/indices "
                      "первых пяти prompt. Выбраны последние завершённые серии recheck5 (5) и awake (100), обе с caffeinate -di. "
                      "Политика wired_memory может различаться между backend, но сохранена при повторе каждого backend. "
                      "Питание и температура не фиксируются caffeinate; их доступные снимки включены в JSON."]
            lines += ["", "## Питание в начале → конце каждого прогона", "",
                      "| Backend | Старые100 awake | Повтор5 recheck5 |",
                      "|---|---|---|"]
            for backend, sources in report["sources"].items():
                lines.append(f"| {backend} | {self._power(sources['awake100'])} | {self._power(sources['recheck5'])} |")
            lines += ["", "При различающемся питании delta отражает повтор в других условиях, а не изолированное "
                      "изменение производительности framework. Снимки относятся к началу и концу всего прогона "
                      "и не определяют питание в момент каждого из первых пяти запросов; причинный эффект батареи не измерен."]
        lines += ["", "## Исходные файлы", ""]
        for backend, cohorts in report["sources"].items():
            for cohort, source in cohorts.items():
                path = source.get("files", {}).get("summary.json", {}).get("path", source.get("selected_summary"))
                if path:
                    lines.append(f"- {backend} / {cohort}: [{Path(path).parent.name}]({path})")
        lines += ["", "Файловые SHA256 и причины исключения других запусков доступны в [JSON](recheck-5.json).", ""]
        return "\n".join(lines)

    def build(self):
        report = self.analyze()
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "recheck-5.json", report)
        fields = ["status", "backend", "cohort", "phase", "samples", "mean_tokens_per_second",
                  "sample_stddev_tokens_per_second", "sample_variance_tokens_per_second",
                  "aggregate_tokens_per_second", "delta_percent_vs_old_first5",
                  "summary_path", "summary_sha256", "prompts_path", "prompts_sha256",
                  "samples_path", "samples_sha256", "dataset_sha256", "model_manifest_sha256", "error"]
        with (self.output / "recheck-5.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in report["comparisons"]:
                for cohort in ("old_first5", "recheck5", "awake100"):
                    source = report["sources"][row["backend"]]["recheck5" if cohort == "recheck5" else "awake100"]
                    for phase in ("prefill", "decode"):
                        stats = row[cohort][phase]
                        record = {"status": "completed", "backend": row["backend"], "cohort": cohort,
                                  "phase": phase, "samples": row[cohort]["samples"],
                                  "delta_percent_vs_old_first5": row["delta_percent"][phase] if cohort == "recheck5" else "",
                                  **{key: stats[key] for key in fields[5:9]},
                                  **{key: source[key] for key in ("dataset_sha256", "model_manifest_sha256")}}
                        for prefix, filename in (("summary", "summary.json"), ("prompts", "prompts.json"), ("samples", "samples.jsonl")):
                            record[f"{prefix}_path"] = source["files"][filename]["path"]
                            record[f"{prefix}_sha256"] = source["files"][filename]["sha256"]
                        writer.writerow(record)
            for error in report["errors"]:
                writer.writerow({"status": "failed", "backend": error["backend"],
                                 "cohort": error["cohort"], "error": error["message"]})
        (self.output / "recheck-5.md").write_text(self._markdown(report))
        print(self.output / "recheck-5.md")
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts/runs")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reports")
    arguments = parser.parse_args()
    report = RecheckReport(arguments.runs, arguments.output).build()
    if report["status"] != "completed":
        raise SystemExit(1)
