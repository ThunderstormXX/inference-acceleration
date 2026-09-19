"""Presentation must preserve the measured sequence and its timing boundaries."""
from copy import deepcopy

import pytest

pytest.importorskip("PIL")
from inference_lab.visualization.render import GenerationReplay, ReplayConfig


def trace():
    ids = [1, 2, 99, 99]
    texts = ["Hello", " sea.", "<eos>", "<eos>"]
    baseline = [dict(type="commit", t=(i + 1) / 10, token_ids=[token], output_count=i + 1)
                for i, token in enumerate(ids)]
    mtp = [dict(type="commit", t=.1, token_ids=[1], output_count=1),
           dict(type="draft", t=.12, token_ids=[2, 99], token_texts=[" sea.", "<eos>"]),
           dict(type="commit", t=.15, token_ids=[2, 99], output_count=3,
                accepted_count=2, draft_count=2, round=1),
           dict(type="commit", t=.2, token_ids=[99], output_count=4)]
    return {"prompt": "Say hello to the sea.", "eos_token_ids": [99], "parity": {"equal": True},
            "baseline": {"token_ids": ids, "token_texts": texts, "events": baseline},
            "mtp": {"token_ids": ids[:], "token_texts": texts[:], "events": mtp}}


def test_eos_cap_uses_observed_completion_of_visible_text():
    replay = GenerationReplay(trace(), ReplayConfig())
    assert replay.count == 2
    assert replay.full_text == "Hello sea."
    assert replay.ends == {"baseline": .2, "mtp": .15}
    assert replay._state("mtp", .13)[0] == 1
    assert replay._state("mtp", .13)[3]["token_ids"] == [2, 99]
    assert replay._state("mtp", 100)[0] == 2
    assert replay._state("mtp", 100)[3] is None


@pytest.mark.parametrize("fault", ["parity", "events", "count", "time"])
def test_corrupt_trace_cannot_present_a_false_exact_match(fault):
    data = deepcopy(trace())
    if fault == "parity":
        data["mtp"]["token_ids"][1] = 13
    elif fault == "events":
        data["mtp"]["events"][2]["token_ids"][0] = 13
    elif fault == "count":
        data["mtp"]["events"][2]["output_count"] = 2
    else:
        data["mtp"]["events"][2]["t"] = .01
    with pytest.raises(ValueError):
        GenerationReplay(data, ReplayConfig())


def test_explicit_newlines_preserve_character_offsets_and_rows():
    replay = GenerationReplay(trace(), ReplayConfig())
    text = "First line.\n\nSecond paragraph. End."
    lines = replay._lines(text, 522)
    assert len(lines) == 3
    assert "".join(line for _, line in lines) == text
    for start, line in lines:
        assert text[start:start + len(line)] == line
