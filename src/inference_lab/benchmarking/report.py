"""Build a report from durable measurements; never synthesize missing rates."""
import argparse
import csv
import json
import math
from pathlib import Path
from inference_lab.core.config import ROOT
from inference_lab.core.io import write_json
from .metrics import aggregate


DISPERSION_EXPORT_FIELDS = (
    "sample_variance_tokens_per_second",
    "sample_stddev_tokens_per_second",
    "min_tokens_per_second",
    "max_tokens_per_second",
    "coefficient_of_variation_percent",
)


class BenchmarkReport:
    def __init__(self, runs: Path = ROOT / "artifacts/runs"):
        self.runs = runs

    def build(self):
        summaries = [(p, json.loads(p.read_text())) for p in sorted(self.runs.glob("*/summary.json"))]
        records = []
        for summary_path, summary in summaries:
            config = summary["config"]
            complete = (summary["status"] == "completed"
                        and summary.get("completed_samples", 0) == config["count"])
            metrics = self._metrics_for_report(summary_path, summary, complete)
            records.append({
                "backend": config["backend"], "label": config["label"],
                "status": summary["status"], "samples": summary.get("completed_samples", 0),
                "requested_samples": config["count"], "max_new_tokens": config["max_new_tokens"],
                "prompt_mode": config["prompt_mode"], "kv_bits": config["kv_bits"],
                "user_initiated": config.get("user_initiated", False),
                "wired_memory": config.get("wired_memory", False),
                "caffeinate_assertions": summary.get("environment", {}).get("caffeinate_assertions"),
                "prefill_mean_tok_s": metrics.get("prefill", {}).get("mean_tokens_per_second") if complete else None,
                "decode_mean_tok_s": metrics.get("decode", {}).get("mean_tokens_per_second") if complete else None,
                "prefill_aggregate_tok_s": metrics.get("prefill", {}).get("aggregate_tokens_per_second") if complete else None,
                "decode_aggregate_tok_s": metrics.get("decode", {}).get("aggregate_tokens_per_second") if complete else None,
                "prompt_tokens_sha256": summary.get("prompt_tokens_sha256"),
                "dataset_sha256": summary.get("dataset_sha256"),
                "model_manifest_sha256": summary.get("model_manifest_sha256"),
                "directory": summary["run_directory"], "error": summary.get("error", ""),
            })
            for phase in ("prefill", "decode"):
                phase_metrics = metrics.get(phase, {})
                for field in DISPERSION_EXPORT_FIELDS:
                    value = phase_metrics.get(field) if complete else None
                    if summary.get("completed_samples", 0) < 2 and field in (
                        "sample_variance_tokens_per_second",
                        "sample_stddev_tokens_per_second",
                        "coefficient_of_variation_percent",
                    ):
                        value = None
                    records[-1][f"{phase}_{field}"] = value
        output = ROOT / "artifacts/reports"
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "comparison.json", records)
        if records:
            with (output / "comparison.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        lines = ["# Результаты локальных запусков", "",
                 "Показаны арифметическое среднее ± выборочное стандартное отклонение tok/s по запросам (SD, делитель N−1). Для одного запроса SD не определено; при отсутствии исходных измерений неизвестный разброс обозначен «—». В JSON/CSV также сохранены выборочная дисперсия (tok/s)², минимум, максимум, CV (%) и отношение суммы токенов к сумме времени.", "",
                 "Сопоставимы только запуски с одинаковой моделью, prompt_tokens_sha256, числом новых токенов и режимом prompt_mode. Идентификатор весов model_manifest_sha256 сохранён в JSON/CSV; без него совпадение весов не подтверждено. Длительность prefill включает первый выходной токен; decode считает N−1 последующих токенов.", ""]
        if (output / "recheck-5.md").is_file():
            lines += ["[Повторный замер четырёх framework на пяти примерах](recheck-5.md).", ""]
        primary = [r for r in records if r["status"] == "completed"
                   and r["samples"] == r["requested_samples"] == 100]
        diagnostic = [r for r in records if r not in primary]
        awake = [r for r in primary if r["label"] == "awake" and r["caffeinate_assertions"] == "di"]
        if awake:
            lines += ["## Итоговая серия awake: 100 примеров, caffeinate -di", "",
                      "Условия предыдущих запусков менялись, поэтому они вынесены в историю. Для итогового сравнения используйте результаты внутри серии awake с одинаковой моделью и нагрузкой. Флаги wired_memory, user_initiated и caffeinate_assertions сохранены для каждого запуска в JSON/CSV.", "",
                      "caffeinate предотвращает idle sleep, но не фиксирует питание, частоты GPU или температуру. Снимки состояния сохранены в environment и resources_after каждого summary.json; ограничения конкретной серии описаны в [методике](../../docs/benchmark-methodology.md).", ""]
            lines += self._table(awake)
            history = [r for r in primary if r not in awake]
            if history:
                lines += ["", "## История полных прогонов на 100 примерах", ""]
                lines += self._table(history)
        else:
            lines += ["## Полные прогоны на 100 примерах", ""]
            lines += self._table(primary)
        if diagnostic:
            lines += ["", "## Проверочные и незавершённые запуски", ""]
            lines += self._table(diagnostic)
        lines += ["", "## Подробности запусков", ""]
        for row in records:
            folder = Path(row["directory"])
            relative = Path("../runs") / folder.name / "summary.json"
            lines += [f"- [{folder.name}]({relative}): {row['error'] or row['status']}"]
        lines += ["", "Метрики скорости не оценивают правильность решений. Фиксированная длина генерации игнорирует EOS. Загрузка весов, токенизация и прогрев исключены. Рабочая нагрузка — текстовая часть Qwen3.5; изображения не используются.", "",
                  "Transformers на macOS 15 использует адаптер исходных MLX affine weights и локальную компиляцию shader templates MLX через PyTorch MPS. Готовый Hub Metal4 binary на этой ОС не исполняется. Это измерение совместимого адаптера, а не прямого стандартного from_pretrained. vLLM измерен с выключенным async scheduling для строгого разделения фаз; scheduler входит во время запроса.", ""]
        (output / "comparison.md").write_text("\n".join(lines))
        print(output / "comparison.md")
        return records

    @staticmethod
    def _metrics_for_report(summary_path, summary, complete):
        """Enrich archived metrics from raw requests without modifying sources."""
        saved = summary.get("metrics", {})
        raw_path = summary_path.with_name("samples.jsonl")
        if not complete or not raw_path.is_file():
            return saved
        rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        expected = summary["config"]["count"]
        if len(rows) != expected or ("samples" in saved and saved["samples"] != expected):
            raise ValueError(
                f"{raw_path}: raw/saved sample count does not match completed request count {expected}"
            )
        recomputed = aggregate(rows)
        for phase in ("prefill", "decode"):
            for field in ("mean_tokens_per_second", "aggregate_tokens_per_second"):
                previous = saved.get(phase, {}).get(field)
                if previous is None:
                    continue
                current = recomputed[phase][field]
                if (type(previous) not in (int, float)
                        or not math.isfinite(previous)
                        or not math.isclose(previous, current, rel_tol=1e-9, abs_tol=1e-12)):
                    raise ValueError(
                        f"{summary_path}: archived {phase}.{field}={previous!r} "
                        f"does not match raw measurements ({current!r})"
                    )
                # Preserve the archived rate exactly after checking agreement.
                recomputed[phase][field] = previous
        return recomputed

    @staticmethod
    def _table(rows):
        lines = ["| Backend / вариант | Статус | Записей | Новых токенов/запрос | Prefill tok/s, mean ± SD | Decode tok/s, mean ± SD |",
                 "|---|---|---:|---:|---:|---:|"]
        for row in rows:
            def rate(phase):
                value = row[f"{phase}_mean_tok_s"]
                if value is None:
                    return "—"
                deviation = row[f"{phase}_sample_stddev_tokens_per_second"]
                deviation_text = f"{deviation:.2f}" if deviation is not None else "—"
                return f"{value:.2f} ± {deviation_text}"
            name = row["backend"] + (f" / {row['label']}" if row["label"] else "")
            lines.append(f"| {name} | {row['status']} | {row['samples']}/{row['requested_samples']} | {row['max_new_tokens']} | {rate('prefill')} | {rate('decode')} |")
        return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts/runs")
    BenchmarkReport(parser.parse_args().runs).build()
