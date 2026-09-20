"""Known step functions and raw-evidence checks; no GUI or GPU imports."""
from copy import deepcopy
import math

import pytest

from inference_lab.visualization.trajectory import TokenTrajectory, TrajectoryReport


def make_lane(commits, prefill=1., *, total=None):
    tokens = []
    events = []
    for time, ids in commits:
        tokens.extend(ids)
        events.append({"type": "commit", "t": time, "token_ids": list(ids), "output_count": len(tokens)})
    total = commits[-1][0] + .1 if total is None else total
    return {"token_ids": tokens, "events": events, "prefill_seconds": prefill,
            "decode_seconds": total - prefill, "total_seconds": total}


def pair():
    baseline = TokenTrajectory(make_lane([(1., [1]), (2., [2]), (3., [3]), (4., [4])]))
    method = TokenTrajectory(make_lane([(1., [1]), (1.5, [2, 3]), (3.5, [4])]))
    return baseline, method


def test_known_step_lead_arrivals_and_population_time_dispersion():
    baseline, method = pair()
    assert baseline.arrival_times() == [1., 2., 3., 4.]
    assert method.arrival_times() == [1., 1.5, 1.5, 3.5]
    assert [method.count(t) for t in [-1., 0., .99, 1., 1.49, 1.5, 3.5, 9.]] == [0, 0, 0, 1, 1, 3, 4, 4]
    result = method.compare(baseline)
    # Leads on intervals [1,1.5),[1.5,2),[2,3),[3,3.5): 0,2,1,0.
    assert result["window"]["start_seconds"] == 1.
    assert result["window"]["end_seconds"] == 3.5
    assert result["time_weighted_lead_mean_tokens"] == .8
    assert result["time_weighted_lead_sd_tokens"] == pytest.approx(math.sqrt(.56))
    assert result["lead_min_tokens"] == 0 and result["lead_max_tokens"] == 2
    assert result["time_saved_to_each_token_seconds"] == [0., .5, 1.5, .5]
    assert result["same_output_completion_ratio"] == pytest.approx(4 / 3.5)
    # Twenty-five 100ms increments: +2,-1,-1,+1 and 21 zeroes.
    assert result["lead_increment_sd_per_100ms_tokens"] == pytest.approx(math.sqrt(7 / 25 - (1 / 25) ** 2))
    reversed_result = baseline.compare(method)
    assert reversed_result["time_weighted_lead_mean_tokens"] == -.8
    assert reversed_result["time_weighted_lead_sd_tokens"] == result["time_weighted_lead_sd_tokens"]


def test_completion_and_step_extension_use_last_observed_commit_not_sync():
    baseline, method = pair()
    assert baseline.end == 4. and baseline.total == 4.1
    times, counts = method.step_points(max(baseline.end, method.end))
    assert times == [0., 1., 1.5, 3.5, 4.]
    assert counts == [0, 1, 3, 4, 4]
    assert baseline.step_points(5.) == ([0., 1., 2., 3., 4., 5.], [0, 1, 2, 3, 4, 4])
    assert method.times == [0., 1., 1.5, 3.5]  # Plot extension never edits observations.
    for endpoint in (3., float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            baseline.step_points(endpoint)


def test_different_output_ids_keep_count_lead_but_never_claim_same_token_latency():
    baseline, _ = pair()
    different = TokenTrajectory(make_lane([(1., [1]), (1.5, [2, 99]), (3.5, [4])]))
    result = different.compare(baseline)
    assert result["exact_token_parity"] is False
    assert result["same_output_completion_ratio"] is None
    assert result["time_saved_to_each_token_seconds"] is None
    assert result["time_weighted_lead_mean_tokens"] == .8


@pytest.mark.parametrize("mutation", [
    lambda lane: lane["events"][1].update(output_count=3),
    lambda lane: lane["events"][1].update(output_count=-1),
    lambda lane: lane["events"][1].update(output_count=True),
    lambda lane: lane["events"][1].update(output_count=1),
    lambda lane: lane["events"][1].update(token_ids=[]),
    lambda lane: lane["events"][1].update(token_ids=[-1]),
    lambda lane: lane["events"][1].update(token_ids=[True]),
    lambda lane: lane["events"][1].update(token_ids=[1.]),
    lambda lane: lane["events"][1].update(token_ids=[999]),
    lambda lane: lane.update(token_ids=[1, 2, 3, 999]),
    lambda lane: lane.update(token_ids=[]),
    lambda lane: lane.update(events=[]),
    lambda lane: lane["events"][1].update(type="unknown"),
    lambda lane: lane["events"][1].update(t=-1),
    lambda lane: lane["events"][1].update(t=float("nan")),
    lambda lane: lane["events"][1].update(t=float("inf")),
    lambda lane: lane["events"][1].update(t=True),
    lambda lane: lane["events"][1].update(t=.5),
    lambda lane: lane["events"][-1].update(t=5.),
    lambda lane: lane["events"][0].update(t=1.1),
    lambda lane: lane.update(prefill_seconds=-1),
    lambda lane: lane.update(prefill_seconds=float("nan")),
    lambda lane: lane.update(decode_seconds=-1),
    lambda lane: lane.update(total_seconds=4.2),
    lambda lane: lane.update(total_seconds=float("inf")),
])
def test_corrupt_counts_ids_and_phase_boundaries_fail_closed(mutation):
    lane = make_lane([(1., [1]), (2., [2]), (3., [3]), (4., [4])])
    mutation(lane)
    with pytest.raises(ValueError):
        TokenTrajectory(lane)


def test_prefill_cannot_commit_multiple_tokens_and_common_window_must_exist():
    with pytest.raises(ValueError, match="exactly the first"):
        TokenTrajectory(make_lane([(1., [1, 2]), (2., [3])]))
    lane = TokenTrajectory(make_lane([(1., [1])]))
    with pytest.raises(ValueError, match="common decode"):
        lane.compare(lane)


def rejection_trace():
    pieces = [f" p{i}" for i in range(1, 19)]
    mtp = make_lane([(float(index), [index]) for index in range(1, 16)], total=18.)
    mtp["token_ids"] += [20, 30, 40]
    mtp["token_texts"] = pieces
    for round_id, before, proposals, accepted, emitted in [(1, 15, [10, 11], 0, [20]), (2, 16, [30, 31], 1, [30, 40])]:
        stamp = 15. + round_id
        context = "".join(pieces[:before])
        proposal_text = [" α", " β"] if round_id == 1 else [" accepted", " rejected"]
        mtp["events"].append({"type": "draft", "t": stamp, "token_ids": proposals, "output_count": before,
                              "round": round_id, "context_text": context, "token_texts": proposal_text})
        mtp["events"].append({"type": "commit", "t": stamp + .2, "token_ids": emitted,
                              "output_count": before + len(emitted), "round": round_id,
                              "draft_count": 2, "accepted_count": accepted, "proposed_token_ids": list(proposals),
                              "rejected_token_ids": proposals[accepted:], "context_text": context,
                              "token_texts": [" correction"] if round_id == 1 else [" accepted", " target"]})
    return {"baseline": make_lane([(1., [1]), (18., list(range(2, 19)))]), "mtp": mtp}


def test_rejection_zero_discards_suffix_while_one_rejects_second_proposal():
    trace = rejection_trace()
    examples = TrajectoryReport(trace).rejection_examples()
    assert [row["accepted_count"] for row in examples] == [0, 1]
    assert examples[0]["proposal_statuses"] == ["rejected", "discarded_suffix"]
    assert examples[1]["proposal_statuses"] == ["accepted", "rejected"]
    assert examples[0]["committed_origins"] == ["target_token"]
    assert examples[1]["committed_origins"] == ["accepted_draft", "target_token"]
    assert examples[0]["proposal_texts"] == [" α", " β"]
    assert examples[1]["committed_texts"] == [" accepted", " target"]
    assert examples[0]["output_count_before"] == 15
    assert examples[0]["context"] == "".join(trace["mtp"]["token_texts"][3:15])
    assert examples[0]["context_token_count"] == 12
    assert examples[0]["context_token_ids"] == list(range(4, 16))
    assert examples[0]["confidence"] is None


def test_context_preserves_proposals_stable_unicode_decode():
    trace = rejection_trace()
    draft = trace["mtp"]["events"][15]
    original = draft["context_text"]
    # Contextual decoding can shorten the final piece before completing byte fragments.
    draft["context_text"] = original[:-1]
    example = TrajectoryReport(trace).rejection_examples()[0]
    assert example["context"] == "".join(trace["mtp"]["token_texts"][3:15])[:-1]
    assert example["context_selection"] == "last_12_generated_token_pieces"


def test_missing_token_pieces_use_last_160_actual_context_characters():
    trace = rejection_trace()
    trace["mtp"].pop("token_texts")
    text = "начало " + "пример " * 50
    trace["mtp"]["events"][15]["context_text"] = text
    example = TrajectoryReport(trace).rejection_examples()[0]
    assert example["context"] == text[-160:]
    assert example["context_selection"] == "last_160_characters_fallback"
    assert example["context_token_count"] is None


@pytest.mark.parametrize("mutation", [
    lambda lane: lane["events"][15].update(output_count=14),
    lambda lane: lane["events"][15].update(t=14.),
    lambda lane: lane["events"][15].update(token_ids=[True, 11]),
    lambda lane: lane["events"][16].update(accepted_count=1),
    lambda lane: lane["events"][16].update(rejected_token_ids=[11]),
    lambda lane: lane["events"][16].update(round=99),
    lambda lane: lane["events"][18].update(token_ids=[999, 40]),
    lambda lane: lane["events"].pop(),
])
def test_rejection_cards_cannot_hide_corrupt_draft_verification_evidence(mutation):
    trace = rejection_trace()
    mutation(trace["mtp"])
    with pytest.raises(ValueError):
        TrajectoryReport(trace)


def test_confidence_is_preserved_in_json_and_concise_card_header():
    trace = rejection_trace()
    confidence = {"p1": .5, "p2": .6, "confidence_product": .3, "decision": "verify"}
    trace["mtp"]["events"][15]["confidence"] = deepcopy(confidence)
    trace["mtp"]["events"][16]["confidence"] = deepcopy(confidence)
    example = TrajectoryReport(trace).rejection_examples()[0]
    assert example["confidence"] == confidence
    assert "p1=0.5 · p2=0.6 · product=0.3" in TrajectoryReport._rejection_header(example)
    trace["mtp"]["events"][16]["confidence"]["p1"] = .7
    with pytest.raises(ValueError, match="evidence differs"):
        TrajectoryReport(trace).rejection_examples()


def test_confidence_can_be_carried_only_on_the_commit():
    trace = rejection_trace()
    trace["mtp"]["events"][16]["confidence"] = {"p1": .2, "p2": .5, "confidence_product": .1, "decision": "verify"}
    assert TrajectoryReport(trace).rejection_examples()[0]["confidence"]["confidence_product"] == .1


def test_explicit_shared_window_uses_same_exact_step_arithmetic():
    baseline, method = pair()
    result = method.compare(baseline, window=(1.5, 3.))
    # On this clipped window the method leads by2 for0.5s, then1 for1s.
    assert result["time_weighted_lead_mean_tokens"] == pytest.approx(4 / 3)
    assert result["time_weighted_lead_sd_tokens"] == pytest.approx(math.sqrt(2 / 9))
    assert result["window"]["start_seconds"] == 1.5
    assert result["window"]["end_seconds"] == 3.
    assert result["time_saved_to_each_token_seconds"] == [0., .5, 1.5, .5]
    assert method.compare(baseline)["window"]["start_seconds"] == 1.


@pytest.mark.parametrize("window", [(0., 3.), (2., 4.), (2., 2.), (3., 2.), (1., float("nan")), [True, 3.], (1.,)])
def test_explicit_window_cannot_escape_any_lanes_decode_interval(window):
    baseline, method = pair()
    with pytest.raises(ValueError, match="Explicit comparison window"):
        method.compare(baseline, window=window)


def test_absence_of_rejected_rounds_needs_no_empty_plot(tmp_path):
    assert TrajectoryReport.draw_rejections([], tmp_path / "unused.png") is None
    assert not (tmp_path / "unused.png").exists()
