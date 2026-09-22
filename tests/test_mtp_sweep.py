"""CPU-only checks for paired experiment accounting and AR dispatch."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from inference_lab.backends.apple.mlx_backend import MLXBackend
from inference_lab.backends.apple.speculative.mtp_backend import MTPBackend
from inference_lab.optimizations.mtp_sweep import (
    SharedTargetBaseline, SweepConfig, compact_summary, paired_plan, token_comparison,
)


@pytest.mark.parametrize("kwargs", [
    {"tokens": True}, {"tokens": 0}, {"tokens": 513}, {"prompt_count": 0},
    {"repeats": 0}, {"block_sizes": ()}, {"block_sizes": (3, 3)},
    {"block_sizes": (True,)}, {"block_sizes": (6,)}, {"trace": 1},
])
def test_bounded_config_rejects_invalid_requests(kwargs):
    with pytest.raises(ValueError):
        SweepConfig(**kwargs)


def test_plan_balances_pair_order_for_every_prompt_and_depth():
    config = SweepConfig()
    rows = paired_plan(config)
    assert len(rows) == 48
    for p in range(3):
        for k in (2, 3, 4, 5):
            orders = [[r["mode"] for r in rows if r["prompt_index"] == p
                       and r["paired_block_size"] == k and r["repeat"] == repeat]
                      for repeat in (0, 1)]
            assert set(orders[0]) == {"ar", "mtp"}
            assert orders[0] == list(reversed(orders[1]))
    assert [r["block_size"] for r in rows if r["mode"] == "mtp" and r["prompt_index"] == 0
            and r["repeat"] == 1] == [5, 4, 3, 2]


def test_shared_baseline_calls_ar_measure_and_resets_rotary_state(monkeypatch):
    mtp = MTPBackend("unused-target", "unused-draft")
    mtp._model = SimpleNamespace(_position_ids=123, _rope_deltas=456)
    mtp._mx = object()
    mtp._tokenizer = object()
    mtp._make_prompt_cache = object()
    mtp._model_config = {"model_type": "qwen3_5"}
    ar = SharedTargetBaseline(mtp)
    monkeypatch.setattr("inference_lab.backends.apple.mlx_backend.RecommendedWiredMemory", lambda mx: nullcontext())
    monkeypatch.setattr(MLXBackend, "_measure", lambda self, prompt, count: ("ar", self._model, prompt, count))
    monkeypatch.setattr(MTPBackend, "_measure", lambda *args: pytest.fail("AR dispatched into MTP"))
    assert ar.measure([1, 2], 16) == ("ar", mtp._model, [1, 2], 16)
    assert ar._model._position_ids is None and ar._model._rope_deltas is None
    assert ar._tokenizer is mtp._tokenizer
    assert ar._make_prompt_cache is mtp._make_prompt_cache
    assert ar.wired_memory is True


def row(mode, seconds, *, ids=None, role="paired", pair_id="p0-k3", prompt_index=0):
    ids = [1, 2, 3, 4] if ids is None else ids
    return {"mode": mode, "block_size": 3 if mode == "mtp" else 1,
            "prompt_index": prompt_index, "repeat": 0, "pair_id": pair_id, "role": role,
            "sleep": {"sleep_detected": False},
            "measurement": {"generated_token_ids": ids, "decode_tokens": len(ids) - 1,
                            "decode_seconds": seconds, "prefill_seconds": 0.1,
                            "draft_acceptance_rate": 0.5, "speculative_rounds": 2}}


def report(rows):
    return {"status": "running", "protocol": {"expected_runs": 4, "block_sizes": [3]}, "runs": rows}


def test_summary_handles_mtp_before_its_first_baseline_without_false_success():
    data = report([row("mtp", 1)])
    assert compact_summary(data)["all_token_ids_match"] is None
    data["runs"].append(row("ar", 2))
    summary = compact_summary(data)
    assert summary["all_token_ids_match"] is True
    assert summary["pairs"][0]["order"] == ["mtp", "ar"]
    assert summary["pairs"][0]["decode_speed_ratio"] == 2
    assert summary["by_mode"]["mtp_k3"]["paired_speed_ratio"]["mean"] == 2


def test_brackets_excluded_from_throughput_and_expose_drift():
    data = report([row("ar", 4, role="bracket", pair_id=None), row("ar", 2),
                   row("mtp", 1), row("ar", 2, role="bracket", pair_id=None)])
    summary = compact_summary(data)
    assert summary["by_mode"]["ar_k1"]["tok_s"]["n"] == 1
    assert summary["by_mode"]["ar_k1"]["pooled_tok_s"] == 1.5
    assert summary["drift_bracket"]["after_over_before"] == 2


def test_parity_checks_ar_drift_and_length_mismatch():
    data = report([row("ar", 2, role="bracket"), row("ar", 2, ids=[1, 9, 3, 4]),
                   row("mtp", 1, ids=[1, 9, 3, 4])])
    summary = compact_summary(data)
    assert summary["all_token_ids_match"] is False
    assert summary["pairs"][0]["pair_token_ids_match"] is True
    assert data["runs"][1]["first_mismatch_index"] == 1
    assert token_comparison([1, 2], [1, 2, 3]) == {"matches_baseline": False, "first_mismatch_index": 2}


def test_unmatched_pairs_and_empty_report_are_not_success():
    assert compact_summary(report([]))["all_token_ids_match"] is None
    summary = compact_summary(report([row("ar", 1)]))
    assert summary["pairs"] == []


def test_stock_draft_plan_has_two_matched_pairs_and_reverses_order():
    from inference_lab.optimizations.mtp_sweep import stock_comparison_plan
    plan = stock_comparison_plan(SweepConfig(prompt_count=1, repeats=2, block_sizes=(3,)))
    assert len(plan) == 8
    assert [r["mode"] for r in plan] == ["ar", "mtp-stock", "ar", "mtp", "mtp", "ar", "mtp-stock", "ar"]
    for pair_id in {r["pair_id"] for r in plan}:
        group = [r for r in plan if r["pair_id"] == pair_id]
        assert len(group) == 2 and sum(r["mode"] == "ar" for r in group) == 1


def test_summary_keeps_stock_mtp_and_shortlist_comparisons_separate():
    control = row("mtp", 1.5, pair_id="stock")
    control["mode"] = "mtp-stock"
    data = report([row("ar", 2, pair_id="stock"), control,
                   row("ar", 2.1, pair_id="short"), row("mtp", 1, pair_id="short")])
    result = compact_summary(data)
    assert result["by_mode"]["mtp-stock_k3"]["paired_speed_ratio"]["mean"] == pytest.approx(2/1.5)
    assert result["by_mode"]["mtp_k3"]["paired_speed_ratio"]["mean"] == 2.1
    assert result["shortlist_over_stock"]["mean"] == 1.5
    assert result["draft_comparisons"][0]["token_ids_match"] is True
