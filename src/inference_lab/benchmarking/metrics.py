"""Summarize per-request rates; sample dispersion uses the N-1 denominator.

Variance is measured in (tokens/s)^2, standard deviation and range in tokens/s,
and coefficient of variation in percent. Sample dispersion is undefined for N=1.
"""
from statistics import mean, median, stdev, variance
import math


DISPERSION_FIELDS = frozenset({
    "sample_variance_tokens_per_second",
    "sample_stddev_tokens_per_second",
    "min_tokens_per_second",
    "max_tokens_per_second",
    "coefficient_of_variation_percent",
})


def validate_measurement(row: dict) -> None:
    for key in ("prefill_seconds", "decode_seconds"):
        if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] <= 0:
            raise ValueError(f"{key} must be a positive finite duration")
    for key in ("prompt_tokens", "generated_tokens", "decode_tokens"):
        if type(row.get(key)) is not int or row[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if row["decode_tokens"] != row["generated_tokens"] - 1:
        raise ValueError("Decode must exclude the first token produced by prefill")


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    result = {"samples": len(rows)}
    for row in rows:
        validate_measurement(row)
    for phase, token_key in (("prefill", "prompt_tokens"), ("decode", "decode_tokens")):
        durations = [row[f"{phase}_seconds"] for row in rows]
        if any(t <= 0 for t in durations):
            raise ValueError("Phase durations must be positive")
        counts = [row[token_key] for row in rows]
        rates = [n / t for n, t in zip(counts, durations)]
        average = mean(rates)
        sample_variance = variance(rates) if len(rates) > 1 else None
        sample_stddev = stdev(rates) if len(rates) > 1 else None
        result[phase] = {
            "mean_tokens_per_second": average,
            "median_tokens_per_second": median(rates),
            "aggregate_tokens_per_second": sum(counts) / sum(durations),
            "total_tokens": sum(counts), "total_seconds": sum(durations),
            "sample_variance_tokens_per_second": sample_variance,
            "sample_stddev_tokens_per_second": sample_stddev,
            "min_tokens_per_second": min(rates),
            "max_tokens_per_second": max(rates),
            "coefficient_of_variation_percent": 100.0 * (sample_stddev / average) if sample_stddev is not None else None,
        }
    result["generated_tokens"] = sum(row["generated_tokens"] for row in rows)
    result["peak_memory_gb"] = max(row.get("peak_memory_gb", 0) for row in rows)
    return result
