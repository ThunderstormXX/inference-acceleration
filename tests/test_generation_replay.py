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


def _decode_gif(path):
    from PIL import Image, ImageSequence
    with Image.open(path) as gif:
        return [(frame.convert("RGB").tobytes(), frame.info["duration"])
                for frame in ImageSequence.Iterator(gif)]


@pytest.mark.parametrize("palette_size", [2, 160, 256])
def test_transparent_deltas_preserve_resets_padding_and_input_images(tmp_path, monkeypatch, palette_size):
    from PIL import Image, GifImagePlugin
    from inference_lab.visualization.render import StreamingGifWriter
    # Distinct colors include dark values: checking palette luminance instead
    # of indexed pixel differences would incorrectly make some resets vanish.
    palette = [value for i in range(palette_size) for value in (i, (i * 17) % 256, (i * 71) % 256)]
    first = Image.frombytes("P", (17, 13), bytes((i % 2) for i in range(17 * 13)))
    first.putpalette(palette)
    second = first.copy()
    second.putpixel((1, 1), 1 - first.getpixel((1, 1)))
    second.putpixel((15, 11), 1 - first.getpixel((15, 11)))
    third = second.copy()
    third.putpixel((1, 1), first.getpixel((1, 1)))  # Erase an old changed pixel.
    third.putpixel((13, 9), 1 - first.getpixel((13, 9)))
    originals = [(frame.tobytes(), frame.getpalette()) for frame in (first, second, third)]
    encoded_options = []
    getdata = GifImagePlugin.getdata

    def capture(frame, **options):
        encoded_options.append((frame.copy(), options.copy()))
        return getdata(frame, **options)

    monkeypatch.setattr(GifImagePlugin, "getdata", capture)
    target = tmp_path / "transparent.gif"
    StreamingGifWriter(target, 20).write([first, first.copy(), second, third, third.copy()])
    assert _decode_gif(target) == [(frame.convert("RGB").tobytes(), duration)
                                  for frame, duration in ((first, 40), (second, 20), (third, 40))]
    assert [(frame.tobytes(), frame.getpalette()) for frame in (first, second, third)] == originals
    assert "transparency" not in encoded_options[0][1]
    for crop, options in encoded_options[1:]:
        assert options["disposal"] == 1 and options["transparency"] == 255
        assert crop.histogram()[255] > 0  # Actually replaced unchanged pixels.
    assert target.read_bytes()[10] & 7 == 7  # Global color table has 256 entries.


def test_occupied_transparency_index_falls_back_without_losing_color(tmp_path, monkeypatch):
    from PIL import Image, GifImagePlugin
    from inference_lab.visualization.render import StreamingGifWriter
    first = Image.frombytes("P", (16, 16), bytes(range(256)))
    first.putpalette([value for i in range(256) for value in (i, 255 - i, (i * 31) % 256)])
    second = first.copy()
    second.putpixel((0, 0), 255)  # A changed pixel must stay visible, not transparent.
    second.putpixel((15, 15), 0)
    third = second.copy()
    third.putpixel((0, 0), 0)  # Transparency can safely resume for a later rectangle.
    observed = []
    getdata = GifImagePlugin.getdata

    def capture(frame, **options):
        observed.append(options.copy())
        return getdata(frame, **options)

    monkeypatch.setattr(GifImagePlugin, "getdata", capture)
    target = tmp_path / "occupied.gif"
    StreamingGifWriter(target, 50).write([first, second, third])
    assert _decode_gif(target) == [(frame.convert("RGB").tobytes(), 50) for frame in (first, second, third)]
    assert "transparency" not in observed[1]
    assert observed[2]["transparency"] == 255


def test_transparent_delta_duplicate_duration_can_exceed_gif_delay_limit(tmp_path):
    from PIL import Image
    from inference_lab.visualization.render import StreamingGifWriter
    first = Image.new("P", (8, 8), 0)
    first.putpalette([0, 0, 0, 255, 255, 255])
    second = first.copy()
    second.putpixel((1, 1), 1)
    second.putpixel((6, 6), 1)
    target = tmp_path / "long-delay.gif"
    StreamingGifWriter(target, 655350).write([first, second, second.copy()])
    decoded = _decode_gif(target)
    assert decoded == [(frame.convert("RGB").tobytes(), 655350) for frame in (first, second, second)]
    assert sum(duration for _, duration in decoded) == 3 * 655350


def test_sparse_changes_inside_large_rectangle_avoid_reencoding_background(tmp_path):
    from random import Random
    from PIL import Image, GifImagePlugin
    from inference_lab.visualization.render import StreamingGifWriter
    rng = Random(17)
    first = Image.frombytes("P", (64, 64), bytes(rng.randrange(160) for _ in range(64 * 64)))
    first.putpalette([value for i in range(256) for value in (i, (i * 17) % 256, (i * 71) % 256)])
    frames = [first]
    for step in (1, 2):
        frame = first.copy()
        frame.putpixel((0, 0), (first.getpixel((0, 0)) + step) % 160)
        frame.putpixel((63, 63), (first.getpixel((63, 63)) + step) % 160)
        frames.append(frame)
    # Both changed corners make the existing opaque delta bbox the full canvas.
    opaque = tmp_path / "opaque.gif"
    with opaque.open("wb") as stream:
        for block in GifImagePlugin.getheader(first.copy(), info={"loop": 0, "optimize": False})[0]:
            stream.write(block)
        for frame in frames:
            for block in GifImagePlugin.getdata(frame.copy(), duration=20, disposal=1):
                stream.write(block)
        stream.write(b";")
    target = tmp_path / "sparse.gif"
    StreamingGifWriter(target, 20).write(frames)
    assert _decode_gif(target) == _decode_gif(opaque)
    assert target.stat().st_size < opaque.stat().st_size * 0.65
