"""Causality, independent suffix oracle, and exact replay-accounting checks."""
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

from inference_lab.optimizations.ngram import (
    LongestSuffixProposer, NgramFeasibilityAnalyzer, accepted_prefix,
)


def naive(history, width, min_match=2, max_match=16):
    for length in range(min(len(history), max_match), min_match - 1, -1):
        suffix = history[-length:]
        for end in range(len(history) - 1, length - 1, -1):
            if history[end - length:end] == suffix:
                return tuple(history[end:min(end + width, len(history))])
    return ()


def test_indexed_proposer_matches_independent_causal_search():
    rng = random.Random(3242)
    proposer = LongestSuffixProposer(min_match=2, max_match=8)
    history = []
    for _ in range(350):
        history.append(rng.randrange(6))
        proposer.commit([history[-1]])
        for width in (1, 2, 4):
            result = proposer.propose(width)
            assert result.token_ids == naive(history, width, 2, 8)
            if result.token_ids:
                assert result.source_start < result.source_end <= len(history)


def test_proposer_never_extrapolates_unknown_tail_or_mutates_on_query():
    proposer = LongestSuffixProposer(min_match=1, max_match=3)
    proposer.commit([7, 7])
    assert proposer.propose(4).token_ids == (7,)
    assert proposer.propose(4).token_ids == (7,)
    assert proposer.committed_tokens == 2
    proposer.commit([9])
    assert proposer.propose(4).token_ids == ()


def test_longest_match_and_latest_occurrence_take_priority():
    proposer = LongestSuffixProposer(min_match=2, max_match=4)
    proposer.commit([1, 2, 3, 4, 5, 1, 2, 9, 1, 2])
    assert proposer.propose(2).token_ids == (9, 1)
    proposer.commit([3, 4])
    assert proposer.propose(2).token_ids == (5, 1)


def test_future_change_cannot_change_proposal_at_unchanged_prefix():
    analyzer = NgramFeasibilityAnalyzer(width=2, min_match=2, max_match=4)
    first = analyzer.analyze([1, 2, 3, 4], [1, 2, 3, 4], mode="all_positions")
    second = analyzer.analyze([1, 2, 3, 4], [1, 2, 9, 9], mode="all_positions")
    a = first["examples"]["accepted"][0]
    b = second["examples"]["rejected_first"][0]
    assert a["output_position"] == b["output_position"] == 2
    assert a["proposal_token_ids"] == b["proposal_token_ids"] == [3, 4]
    assert a["accepted_prefix_length"] == 2 and b["accepted_prefix_length"] == 0


def test_round_accounting_has_bonus_and_fixed_budget_tail():
    analyzer = NgramFeasibilityAnalyzer(width=2, min_match=2)
    # Seed=1, then AR 2, then copy3,4 and bonus5: 4 decode tokens in2 rounds.
    result = analyzer.analyze([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], mode="simulated_rounds")
    assert result["decode_tokens"] == 4
    assert result["visited_positions"] == 2
    assert result["draft_opportunities"] == 1
    assert result["accepted_prefix_tokens"] == 2
    assert result["tokens_per_simulated_round"] == 2
    tail = analyzer.analyze([1, 2, 3, 4], [1, 2, 3, 4], mode="simulated_rounds")
    assert tail["visited_positions"] == 2 and tail["decode_tokens"] == 3


def test_prefix_acceptance_stops_at_first_error():
    assert accepted_prefix([1, 9, 3], [1, 2, 3]) == 1
    assert accepted_prefix([9, 2, 3], [1, 2, 3]) == 0


def test_aggregate_uses_raw_counts_not_mean_of_rates():
    analyzer = NgramFeasibilityAnalyzer(width=1, min_match=1)
    a = analyzer.analyze([1, 1], [1, 1], mode="all_positions")
    b = analyzer.analyze([7], [8, 9, 10, 11], mode="all_positions")
    result = analyzer.aggregate([a, b])
    assert result["opportunity_fraction"] == .25
    assert result["opportunity_macro_mean"] == .5
    assert result["accepted_prefix_histogram"] == {"0": 0, "1": 1}


@pytest.mark.parametrize("bad", [[True], [-1], [1.5], [1, "2"]])
def test_bad_ids_fail_without_partial_commit(bad):
    proposer = LongestSuffixProposer()
    with pytest.raises(ValueError):
        proposer.commit(bad)
    assert proposer.committed_tokens == 0


def test_no_gpu_runtime_imports():
    code = ('import sys; from inference_lab.optimizations.ngram import LongestSuffixProposer; '
            'assert not any(n.split(".")[0] in {"mlx", "torch", "mlx_lm", "mlx_vlm"} for n in sys.modules)')
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
