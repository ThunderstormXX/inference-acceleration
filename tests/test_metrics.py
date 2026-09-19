import pytest
import math
from inference_lab.benchmarking.metrics import aggregate
from inference_lab.core.config import BenchmarkConfig


def test_arithmetic_and_aggregate_have_distinct_meanings():
    rows = [dict(prompt_tokens=100, prefill_seconds=1, generated_tokens=11, decode_tokens=10, decode_seconds=1),
            dict(prompt_tokens=100, prefill_seconds=10, generated_tokens=11, decode_tokens=10, decode_seconds=2)]
    result = aggregate(rows)
    assert result["prefill"]["mean_tokens_per_second"] == 55
    assert result["prefill"]["aggregate_tokens_per_second"] == pytest.approx(200/11)
    assert result["decode"]["total_tokens"] == 20
    assert result["generated_tokens"] == 22


def test_need_decode_step():
    with pytest.raises(ValueError):
        BenchmarkConfig("mlx", max_new_tokens=1)


def test_five_request_dispersion_uses_sample_denominator_and_request_rates():
    # Prefill rates are 10,20,30,40,50: squared deviations sum to 1000,
    # so sample variance is 1000/(5-1)=250 (population variance would be 200).
    # Decode rates are 1,2,3,4,5, with sample variance 10/(5-1)=2.5.
    rows = [dict(prompt_tokens=100, prefill_seconds=100 / rate,
                 generated_tokens=11, decode_tokens=10, decode_seconds=100 / rate)
            for rate in (10, 20, 30, 40, 50)]
    result = aggregate(rows)
    for phase, average, variance, minimum, maximum in (
            ("prefill", 30, 250, 10, 50), ("decode", 3, 2.5, 1, 5)):
        values = result[phase]
        assert values["mean_tokens_per_second"] == average
        assert values["sample_variance_tokens_per_second"] == variance
        assert values["sample_stddev_tokens_per_second"] == pytest.approx(math.sqrt(variance))
        assert values["coefficient_of_variation_percent"] == pytest.approx(100 * math.sqrt(variance) / average)
        assert values["min_tokens_per_second"] == minimum
        assert values["max_tokens_per_second"] == maximum


def test_dispersion_does_not_change_existing_aggregate_values():
    rows = [dict(prompt_tokens=100, prefill_seconds=1, generated_tokens=11, decode_tokens=10, decode_seconds=1),
            dict(prompt_tokens=100, prefill_seconds=10, generated_tokens=11, decode_tokens=10, decode_seconds=2)]
    result = aggregate(rows)
    assert result["samples"] == 2
    assert result["generated_tokens"] == 22
    assert result["peak_memory_gb"] == 0
    for phase, expected in {
        "prefill": {"mean_tokens_per_second": 55, "median_tokens_per_second": 55,
                    "aggregate_tokens_per_second": 200 / 11, "total_tokens": 200, "total_seconds": 11},
        "decode": {"mean_tokens_per_second": 7.5, "median_tokens_per_second": 7.5,
                   "aggregate_tokens_per_second": 20 / 3, "total_tokens": 20, "total_seconds": 3},
    }.items():
        assert {key: result[phase][key] for key in expected} == expected


def test_single_request_has_undefined_sample_dispersion_and_valid_range():
    row = dict(prompt_tokens=100, prefill_seconds=2, generated_tokens=11, decode_tokens=10, decode_seconds=4)
    result = aggregate([row])
    for phase, rate in (("prefill", 50), ("decode", 2.5)):
        assert result[phase]["sample_variance_tokens_per_second"] is None
        assert result[phase]["sample_stddev_tokens_per_second"] is None
        assert result[phase]["coefficient_of_variation_percent"] is None
        assert result[phase]["min_tokens_per_second"] == rate
        assert result[phase]["max_tokens_per_second"] == rate


def test_equal_request_rates_have_zero_dispersion():
    rows = [dict(prompt_tokens=100, prefill_seconds=2, generated_tokens=11, decode_tokens=10, decode_seconds=4)
            for _ in range(5)]
    result = aggregate(rows)
    for phase in ("prefill", "decode"):
        assert result[phase]["sample_variance_tokens_per_second"] == 0
        assert result[phase]["sample_stddev_tokens_per_second"] == 0
        assert result[phase]["coefficient_of_variation_percent"] == 0


def test_empty_measurements_still_have_no_aggregates():
    assert aggregate([]) == {}
