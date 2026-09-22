"""Causal CPU proposals from committed tokens; no model or GPU imports.

This module estimates opportunities on existing trajectories. It does not run a
verifier and cannot establish latency gains or numerical parity of GPU kernels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, stdev
from time import perf_counter_ns
from typing import Iterable, Sequence


def _tokens(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(values)
    if any(type(value) is not int or value < 0 for value in result):
        raise ValueError("token IDs must be nonnegative integers")
    return result


@dataclass(frozen=True)
class NgramProposal:
    token_ids: tuple[int, ...]
    matched_suffix_length: int = 0
    source_start: int | None = None
    source_end: int | None = None


class LongestSuffixProposer:
    """Copy the latest earlier continuation of the longest matching suffix.

    Only ``commit`` adds knowledge. An occurrence is indexed once at least one
    following token has been committed. Copied continuations end at the current
    history boundary; even repetitive inputs never extrapolate unknown tokens.
    Querying does not modify history. No proposals are committed speculatively.
    """

    def __init__(self, *, min_match: int = 2, max_match: int = 16):
        if type(min_match) is not int or type(max_match) is not int or not 1 <= min_match <= max_match:
            raise ValueError("require 1 <= min_match <= max_match")
        self.min_match = min_match
        self.max_match = max_match
        self._history: list[int] = []
        self._latest: dict[tuple[int, ...], int] = {}

    @property
    def committed_tokens(self) -> int:
        return len(self._history)

    def commit(self, token_ids: Iterable[int]) -> None:
        # Validate before modifying state so malformed input is atomic.
        for token in _tokens(token_ids):
            boundary = len(self._history)
            self._history.append(token)
            for length in range(self.min_match, min(self.max_match, boundary) + 1):
                self._latest[tuple(self._history[boundary - length:boundary])] = boundary

    def propose(self, max_tokens: int) -> NgramProposal:
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative integer")
        if max_tokens == 0:
            return NgramProposal(())
        for length in range(min(self.max_match, len(self._history)), self.min_match - 1, -1):
            continuation = self._latest.get(tuple(self._history[-length:]))
            if continuation is not None:
                end = min(continuation + max_tokens, len(self._history))
                return NgramProposal(tuple(self._history[continuation:end]), length, continuation, end)
        return NgramProposal(())


def accepted_prefix(proposal: Sequence[int], labels: Sequence[int]) -> int:
    """Labels are used after proposing, never supplied to the proposer."""
    count = 0
    for proposed, actual in zip(proposal, labels):
        if proposed != actual:
            break
        count += 1
    return count


class NgramFeasibilityAnalyzer:
    """Teacher-forced opportunities and causal simulated speculative rounds."""

    def __init__(self, *, width: int, min_match: int = 2, max_match: int = 16):
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        LongestSuffixProposer(min_match=min_match, max_match=max_match)
        self.width, self.min_match, self.max_match = width, min_match, max_match

    def analyze(self, prompt_ids: Sequence[int], output_ids: Sequence[int], *, mode: str) -> dict:
        prompt, output = _tokens(prompt_ids), _tokens(output_ids)
        if not prompt or len(output) < 2:
            raise ValueError("need prompt and at least two output tokens")
        if mode not in {"all_positions", "simulated_rounds"}:
            raise ValueError("unknown replay mode")
        proposer = LongestSuffixProposer(min_match=self.min_match, max_match=self.max_match)
        proposer.commit(prompt + output[:1])  # First generated token belongs to prefill.
        position = 1
        opportunities = offered = accepted = full = full_width = visited = 0
        proposal_ns = 0
        histogram: Counter[int] = Counter()
        match_histogram: Counter[int] = Counter()
        width_histogram: Counter[int] = Counter()
        examples: dict[str, list[dict]] = {"accepted": [], "rejected_first": [], "partial": []}
        while position < len(output):
            start = perf_counter_ns()
            proposal = proposer.propose(min(self.width, len(output) - position))
            proposal_ns += perf_counter_ns() - start
            visited += 1
            count = accepted_prefix(proposal.token_ids, output[position:position + len(proposal.token_ids)])
            if proposal.token_ids:
                opportunities += 1
                offered += len(proposal.token_ids)
                accepted += count
                full += count == len(proposal.token_ids)
                full_width += len(proposal.token_ids) == self.width
                histogram[count] += 1
                match_histogram[proposal.matched_suffix_length] += 1
                width_histogram[len(proposal.token_ids)] += 1
                category = "accepted" if count == len(proposal.token_ids) else "partial" if count else "rejected_first"
                if len(examples[category]) < 3:
                    examples[category].append({
                        "output_position": position,
                        "prefix_token_ids": list((prompt + output[:position])[-12:]),
                        "proposal_token_ids": list(proposal.token_ids),
                        "label_token_ids": list(output[position:position + len(proposal.token_ids)]),
                        "accepted_prefix_length": count,
                        "matched_suffix_length": proposal.matched_suffix_length,
                        "source_start": proposal.source_start, "source_end": proposal.source_end,
                        "available_history_tokens": proposer.committed_tokens,
                    })
            # A real exact verifier would emit the accepted prefix plus one
            # correction/bonus token, subject to the fixed output budget.
            advance = 1 if mode == "all_positions" else min(count + 1, len(output) - position)
            proposer.commit(output[position:position + advance])
            position += advance
        return {
            "mode": mode, "decode_tokens": len(output) - 1, "visited_positions": visited,
            "draft_opportunities": opportunities, "offered_tokens": offered,
            "accepted_prefix_tokens": accepted, "fully_accepted_proposals": full,
            "full_width_proposals": full_width,
            "accepted_prefix_histogram": {str(i): histogram[i] for i in range(self.width + 1)},
            "matched_suffix_histogram": dict(sorted(match_histogram.items())),
            "proposal_length_histogram": dict(sorted(width_histogram.items())),
            "cpu_propose_nanoseconds": proposal_ns,
            "examples": examples,
            **self.ratios(visited, opportunities, offered, accepted, full, histogram.get(0, 0)),
            "tokens_per_simulated_round": (len(output) - 1) / visited if mode == "simulated_rounds" else None,
        }

    @staticmethod
    def ratios(visited: int, opportunities: int, offered: int, accepted: int, full: int, zeros: int) -> dict:
        return {
            "opportunity_fraction": opportunities / visited if visited else None,
            "accepted_fraction_of_offered": accepted / offered if offered else None,
            "mean_accepted_prefix_per_opportunity": accepted / opportunities if opportunities else None,
            "first_token_accuracy_given_opportunity": (opportunities - zeros) / opportunities if opportunities else None,
            "fully_accepted_proposal_fraction": full / opportunities if opportunities else None,
        }

    @classmethod
    def aggregate(cls, records: Sequence[dict]) -> dict:
        if not records or len({row["mode"] for row in records}) != 1:
            raise ValueError("need nonempty results from one replay mode")
        summed = {key: sum(row[key] for row in records) for key in (
            "decode_tokens", "visited_positions", "draft_opportunities", "offered_tokens",
            "accepted_prefix_tokens", "fully_accepted_proposals", "full_width_proposals",
            "cpu_propose_nanoseconds",
        )}
        histogram: Counter[str] = Counter()
        for row in records:
            histogram.update(row["accepted_prefix_histogram"])
        ratios = cls.ratios(summed["visited_positions"], summed["draft_opportunities"],
                            summed["offered_tokens"], summed["accepted_prefix_tokens"],
                            summed["fully_accepted_proposals"], histogram["0"])
        per_chain = [row["opportunity_fraction"] for row in records]
        return {**summed, **ratios, "chains": len(records), "mode": records[0]["mode"],
                "accepted_prefix_histogram": dict(sorted(histogram.items())),
                "opportunity_macro_mean": mean(per_chain),
                "opportunity_chain_sample_sd": stdev(per_chain) if len(per_chain) > 1 else None,
                "cpu_propose_microseconds_per_query": summed["cpu_propose_nanoseconds"] / summed["visited_positions"] / 1000,
                "tokens_per_simulated_round": summed["decode_tokens"] / summed["visited_positions"] if records[0]["mode"] == "simulated_rounds" else None}
