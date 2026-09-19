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


def test_long_replay_scrolls_without_dropping_recorded_tokens():
    data = trace()
    data["eos_token_ids"] = []
    for key in ("baseline", "mtp"):
        data[key]["token_texts"] = [("An exact long reasoning line.\n" * 8) for _ in range(4)]
    replay = GenerationReplay(data, ReplayConfig())
    assert replay.scrolling and replay.count == 4
    assert replay.full_text == "".join(data["baseline"]["token_texts"])
    assert replay.frame(100).size == (1280, 920)


def test_diverging_outputs_have_separate_text_and_no_false_parity(tmp_path):
    data = deepcopy(trace())
    data["mtp"]["token_ids"][1] = 13
    data["mtp"]["events"][2]["token_ids"][0] = 13
    data["mtp"]["token_texts"][1] = " different."
    data["parity"]["equal"] = False
    with pytest.raises(ValueError, match="Output tokens differ"):
        GenerationReplay(data, ReplayConfig())
    replay = GenerationReplay(data, ReplayConfig(allow_mismatch=True, fps=10, intro_seconds=0, outro_seconds=0))
    report = replay.render(tmp_path / "divergence.gif")
    assert replay.lane_texts["mtp"] == "Hello different."
    assert replay.lane_texts["baseline"] == "Hello sea."
    assert report["all_captured_token_ids_equal"] is False
    assert report["speedup_request"] is None


def test_streaming_gif_preserves_delta_pixels_and_duplicate_frame_duration(tmp_path):
    from PIL import Image, ImageSequence
    from inference_lab.visualization.render import StreamingGifWriter
    palette = [0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * (768 - 9)
    first = Image.new("P", (16, 12), 0)
    first.putpalette(palette)
    second = first.copy()
    second.putpixel((4, 7), 1)
    third = second.copy()
    third.putpixel((10, 3), 2)
    target = tmp_path / "stream.gif"
    StreamingGifWriter(target, 50).write(iter([first, first.copy(), second, third, third.copy()]))
    with Image.open(target) as gif:
        frames = [(f.convert("RGB").copy(), f.info["duration"]) for f in ImageSequence.Iterator(gif)]
        assert gif.info["loop"] == 0
    assert [duration for _, duration in frames] == [100, 50, 100]
    for (actual, _), expected in zip(frames, [first, second, third]):
        assert actual.tobytes() == expected.convert("RGB").tobytes()


def test_prefix_preview_keeps_original_timestamps():
    data = trace()
    data["eos_token_ids"] = []
    replay = GenerationReplay(data, ReplayConfig(max_visible_tokens=2))
    assert replay.counts == {"baseline": 2, "mtp": 2}
    assert replay.ends == {"baseline": .2, "mtp": .15}


def test_long_formula_never_spills_into_the_adjacent_lane():
    replay = GenerationReplay(trace(), ReplayConfig())
    text = "\\boxed{" + "1" * 120 + "}"
    lines = replay._lines(text, 522)
    assert len(lines) > 1
    assert "".join(line for _, line in lines) == text
    font = replay.type(replay.font_size, "serif")
    for start, line in lines:
        assert text[start:start + len(line)] == line
        assert font.getlength(line.rstrip()) <= 522


def test_prefix_limit_is_not_reported_as_eos():
    data = trace()
    data["eos_token_ids"] = []
    replay = GenerationReplay(data, ReplayConfig(max_visible_tokens=2))
    assert replay.eos_positions == {"baseline": None, "mtp": None}
    assert replay.stop_reasons == {"baseline": "prefix_limit", "mtp": "prefix_limit"}


def test_gif_fps_must_represent_real_timing_without_rounding_drift():
    with pytest.raises(ValueError, match="divide 100"):
        ReplayConfig(fps=30)
