"""A typography-led GIF replay of observed, unmodified inference event times."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont, ImageChops, GifImagePlugin

from inference_lab.core.config import ROOT


@dataclass(frozen=True)
class ReplayConfig:
    width: int = 1280
    height: int = 920
    fps: int = 25
    playback_rate: float = 1.0
    intro_seconds: float = 1.4
    outro_seconds: float = 2.8
    max_visible_tokens: int | None = None
    allow_mismatch: bool = False

    def __post_init__(self):
        if not math.isfinite(self.playback_rate) or self.playback_rate <= 0:
            raise ValueError("Playback rate must be positive and finite")
        if type(self.fps) is not int or self.fps not in (1, 2, 4, 5, 10, 20, 25, 50, 100):
            raise ValueError("GIF FPS must divide 100 exactly: 1, 2, 4, 5, 10, 20, 25, 50 or 100")
        if self.max_visible_tokens is not None and (type(self.max_visible_tokens) is not int or self.max_visible_tokens < 2):
            raise ValueError("max_visible_tokens must be at least 2")


class StreamingGifWriter:
    """Encode long replays with a shared palette and bounded frame memory.

    Delta rectangles use disposal=1 so previous pixels remain on screen.
    Unchanged pixels inside each rectangle become transparent when palette
    index 255 is unused there; occupied rectangles safely remain opaque.
    Identical frames accumulate duration instead of allocating more images.
    """

    def __init__(self, output: Path, duration_ms: int):
        self.output = Path(output)
        # GIF stores time in centiseconds; report/CLI FPS should fit that grid.
        self.duration_ms = max(10, round(duration_ms / 10) * 10)

    def write(self, frames):
        frames = iter(frames)
        pending = next(frames)
        if pending.mode != "P":
            raise ValueError("Streaming GIF requires a shared indexed palette")
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("wb") as stream:
                # Transparency index 255 must exist even with a short source
                # palette. getheader may mutate its image, so pad a private copy.
                header_frame = pending.copy()
                palette = header_frame.getpalette() or []
                header_frame.putpalette(palette + [0] * (768 - len(palette)))
                header, _ = GifImagePlugin.getheader(header_frame, info={"loop": 0, "optimize": False})
                for block in header:
                    stream.write(block)
                previous = None
                duration = self.duration_ms

                def flush(frame, milliseconds, prior):
                    difference = ImageChops.difference(prior, frame) if prior is not None else None
                    bbox = difference.getbbox() if difference is not None else None
                    if bbox is None:
                        bbox = (0, 0, frame.width, frame.height)
                    crop = frame.crop(bbox)
                    options = {}
                    if difference is not None and crop.histogram()[255] == 0:
                        # Index differences, not palette RGB luminance, decide
                        # which pixels changed. Real resets to the background
                        # remain opaque, while unchanged text need not be LZW
                        # encoded again inside a large delta rectangle.
                        unchanged = difference.crop(bbox).point([255] + [0] * 255, mode="L")
                        crop.paste(255, mask=unchanged)
                        options["transparency"] = 255
                    # A GIF delay is a uint16 number of centiseconds.
                    while milliseconds:
                        part = min(milliseconds, 655350)
                        for block in GifImagePlugin.getdata(crop, offset=bbox[:2], duration=part, disposal=1, **options):
                            stream.write(block)
                        milliseconds -= part

                for frame in frames:
                    if frame.mode != "P" or frame.size != pending.size or frame.getpalette() != pending.getpalette():
                        raise ValueError("All GIF frames must share dimensions and palette")
                    if ImageChops.difference(pending, frame).getbbox() is None:
                        duration += self.duration_ms
                        continue
                    flush(pending, duration, previous)
                    previous, pending, duration = pending, frame, self.duration_ms
                flush(pending, duration, previous)
                stream.write(b";")
            temporary.replace(self.output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


class Typography:
    def __init__(self):
        self.directory = Path("/System/Library/Fonts/Supplemental")
        self.cache = {}

    def __call__(self, size, style="regular"):
        key = (size, style)
        if key not in self.cache:
            name = {"regular": "Arial.ttf", "bold": "Arial Bold.ttf",
                    "serif": "Georgia.ttf", "mono": "Andale Mono.ttf"}[style]
            self.cache[key] = ImageFont.truetype(str(self.directory / name), size)
        return self.cache[key]


def mtp_panel_labels(metadata):
    """Show the recorded draft configuration, preserving legacy demo labels."""
    block = metadata.get("block_size", 3)
    if type(block) is not int or not 2 <= block <= 5:
        raise ValueError("Recorded MTP block size must be in 2..5")
    title = f"MTP · блок {block}"
    subtitle = f"{block - 1} draft-токена → проверка target"
    vocabulary = metadata.get("draft_vocabulary")
    if vocabulary is not None:
        size, total = vocabulary.get("shortlist_size"), vocabulary.get("target_vocab_size")
        if type(size) is not int or type(total) is not int or not 0 < size <= total:
            raise ValueError("Recorded draft vocabulary sizes are invalid")
        title = f"MTP · K={block} · словарь {size:,}".replace(",", " ")
        subtitle = f"До {block - 1} draft-токенов → полный словарь target"
    return title, subtitle


class GenerationReplay:
    BG = "#0b101b"
    PANEL = "#121b2a"
    INK = "#eef2f5"
    MUTED = "#94a7bd"
    BLUE = "#8aaef4"
    TEAL = "#62e5bf"
    GOLD = "#f5c779"
    RED = "#f4939b"
    TRACK = "#263348"

    def __init__(self, trace: dict, config: ReplayConfig):
        actual_equal = trace["baseline"]["token_ids"] == trace["mtp"]["token_ids"]
        if bool(trace.get("parity", {}).get("equal")) != actual_equal:
            raise ValueError("Trace parity flag disagrees with actual token IDs")
        if not actual_equal and not config.allow_mismatch:
            raise ValueError("Output tokens differ; use allow_mismatch to show both actual trajectories")
        self.equal = actual_equal
        self.trace, self.config = trace, config
        self.type = Typography()
        self.ids = trace["baseline"]["token_ids"]
        self.counts, self.lane_prefixes, self.lane_texts, self.ends = {}, {}, {}, {}
        self.eos_positions, self.stop_reasons = {}, {}
        eos = set(trace.get("eos_token_ids", trace.get("metadata", {}).get("eos_token_ids", [])))
        for key in ("baseline", "mtp"):
            lane = trace[key]
            ids = lane["token_ids"]
            if len(lane["token_texts"]) != len(ids):
                raise ValueError("Token text count must match token ID count")
            committed, previous_t = [], -1.0
            for event in lane["events"]:
                if not math.isfinite(event["t"]) or event["t"] < previous_t or event["t"] < 0:
                    raise ValueError("Events must have finite chronological timestamps")
                previous_t = event["t"]
                if event["type"] == "commit":
                    committed.extend(event["token_ids"])
                    if event["output_count"] != len(committed):
                        raise ValueError("Commit output count disagrees with recorded token IDs")
            if committed != ids:
                raise ValueError("Commit events disagree with final generated token IDs")
            eos_position = next((i for i, token in enumerate(ids) if token in eos), None)
            self.eos_positions[key] = eos_position
            count = eos_position if eos_position is not None else len(ids)
            if config.max_visible_tokens is not None:
                count = min(count, config.max_visible_tokens)
            self.stop_reasons[key] = ("first_eos" if eos_position is not None and count == eos_position
                                      else "prefix_limit" if count < len(ids) else "recorded_budget")
            if count < 2:
                raise ValueError("Replay requires at least two visible output tokens per lane")
            self.counts[key] = count
            prefixes = [""]
            for text in lane["token_texts"][:count]:
                prefixes.append(prefixes[-1] + text)
            self.lane_prefixes[key] = prefixes
            self.lane_texts[key] = prefixes[-1]
            self.ends[key] = next(e["t"] for e in lane["events"]
                                  if e["type"] == "commit" and e["output_count"] >= count)
        # Keep the original equal-output convenience attributes for callers.
        self.count = self.counts["baseline"]
        self.prefixes = self.lane_prefixes["baseline"]
        self.full_text = self.lane_texts["baseline"]
        self.end = max(self.ends.values())
        self.font_size = 23
        self.scrolling = any(len(self._lines(text, 522)) > 12 for text in self.lane_texts.values())
        if not self.scrolling:
            for size in (23, 22, 21, 20, 19):
                self.font_size = size
                if all(len(self._lines(text, 522)) <= 12 for text in self.lane_texts.values()):
                    break
        self.layouts = {key: self._lines(text, 522) for key, text in self.lane_texts.items()}
        self.base = self._base()

    def _lines(self, text: str, width: int):
        font = self.type(self.font_size, "serif")
        lines, start = [], 0
        # Preserve every character and its offset, including long formulas with
        # no spaces. A line break is a layout decision, never a text mutation.
        for physical in text.splitlines(keepends=True):
            content = physical.rstrip("\r\n")
            ending = physical[len(content):]
            line = ""
            for match in re.finditer(r"\S+\s*|\s+", content):
                part = match.group()
                if line and font.getlength((line + part).rstrip()) > width:
                    lines.append((start, line))
                    start += len(line)
                    line = ""
                while part and font.getlength(part.rstrip()) > width:
                    lo, hi = 1, len(part)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if font.getlength(part[:mid]) <= width:
                            lo = mid
                        else:
                            hi = mid - 1
                    lines.append((start, part[:lo]))
                    start += lo
                    part = part[lo:]
                line += part
            lines.append((start, line + ending))
            start += len(line + ending)
        return lines or [(0, "")]

    def _text(self, draw, xy, text, size=18, color=None, style="regular", anchor=None):
        draw.text(xy, text, fill=color or self.INK, font=self.type(size, style), anchor=anchor)

    def _base(self):
        image = Image.new("RGB", (self.config.width, self.config.height), self.BG)
        draw = ImageDraw.Draw(image)
        # Restrained background light, fixed across frames to preserve GIF quality.
        self._text(draw, (40, 30), "QWEN3.5 / 9B / APPLE SILICON", 14, self.MUTED, "mono")
        self._text(draw, (40, 57), ("Один текст. Два ритма." if self.equal else "Один промпт. Два пути."), 42, style="bold")
        subtitle = (f"Начало рассуждения · первые {self.count} токенов · native MTP"
                    if self.trace.get("metadata", {}).get("enable_thinking")
                    else "Обычная генерация и speculative decoding с native MTP")
        if self.scrolling:
            subtitle = "Длинная генерация · прокрутка текста · реальные события native MTP"
        self._text(draw, (40, 112), subtitle, 19, self.MUTED)
        draw.rounded_rectangle((40, 155, 1240, 232), radius=14, fill="#172132")
        self._text(draw, (58, 170), "ПРОМПТ", 12, self.GOLD, "bold")
        prompt_lines, line = [], ""
        for word in self.trace.get("prompt", "").split():
            candidate = (line + " " + word).lstrip()
            if line and self.type(17).getlength(candidate) > 1056:
                prompt_lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            prompt_lines.append(line)
        if len(prompt_lines) > 3:
            prompt_lines[2] = prompt_lines[2][:110].rstrip() + "…"
        for i, line in enumerate(prompt_lines[:3]):
            self._text(draw, (158, 165 + 21 * i), line, 17, self.INK if i == 0 else self.MUTED)
        mtp_title, mtp_subtitle = mtp_panel_labels(self.trace.get("metadata", {}))
        for x, title, sub, color in [
            (40, "Обычный decode", "1 токен за шаг большой модели", self.BLUE),
            (654, mtp_title, mtp_subtitle, self.TEAL),
        ]:
            draw.rounded_rectangle((x, 254, x + 586, 790), radius=20, fill=self.PANEL)
            draw.rounded_rectangle((x + 22, 278, x + 26, 330), radius=2, fill=color)
            self._text(draw, (x + 40, 276), title, 26, style="bold")
            self._text(draw, (x + 40, 313), sub, 16, self.MUTED)
            draw.line((x + 24, 351, x + 562, 351), fill=self.TRACK)
        self._text(draw, (40, 882), "Один prompt · greedy · точное сравнение token IDs · последовательные запуски", 14, self.MUTED)
        self._text(draw, (1240, 882), "РЕАЛЬНЫЕ СОБЫТИЯ", 13, self.MUTED, "mono", "ra")
        return image

    def _state(self, key, t):
        events = self.trace[key]["events"]
        commits = [e for e in events if e["type"] == "commit" and e["t"] <= min(t, self.ends[key])]
        limit = self.counts[key]
        count = min(commits[-1]["output_count"], limit) if commits else 0
        last = commits[-1] if commits else None
        previous = min(commits[-2]["output_count"], limit) if len(commits) > 1 else 0
        pending = None
        if key == "mtp" and count < limit:
            proposals = [e for e in events if e["type"] == "draft" and e["t"] <= t]
            if proposals:
                candidate = proposals[-1]
                if last is None or candidate["t"] > last["t"]:
                    pending = candidate
        return count, last, previous, pending

    def _draw_story(self, draw, x, count, last, previous, pending, t, color, key="baseline"):
        prefixes = self.lane_prefixes[key]
        full_text = self.lane_texts[key]
        committed = prefixes[count]
        proposal = ""
        if pending:
            committed = pending.get("context_text", committed)
            proposal = "".join(pending.get("token_texts", []))
            if not proposal:
                proposal = pending.get("text", "")
        text = committed + proposal
        font = self.type(self.font_size, "serif")
        line_height = 30
        # Wrap against final text for stable word positions when possible; draft
        # rejections get their own layout until committed text is restored.
        layout = self.layouts[key] if full_text.startswith(text) else self._lines(text, 522)
        visible_layout = [(start, line) for start, line in layout if start < len(text)]
        window = visible_layout[-12:] if self.scrolling else layout[:12]
        cursor_x, cursor_y = x, 380
        fresh = last is not None and 0 <= t - last["t"] < 0.18
        for row, (start, line) in enumerate(window):
            y = 380 + row * line_height
            visible = text[start:start + len(line)].rstrip("\r\n")
            if not visible:
                if start >= len(text):
                    break
                continue
            if fresh:
                lo = max(0, len(prefixes[previous]) - start)
                hi = min(len(visible), len(committed) - start)
                if hi > lo:
                    left = x + font.getlength(visible[:lo])
                    right = x + font.getlength(visible[:hi])
                    draw.rounded_rectangle((left - 2, y - 2, right + 2, y + 27), radius=4,
                                           fill="#214942" if color == self.TEAL else "#253b62")
            committed_part = visible[:max(0, len(committed) - start)]
            draft_part = visible[len(committed_part):]
            draw.text((x, y), committed_part, font=font, fill=self.INK)
            if draft_part:
                dx = x + font.getlength(committed_part)
                draw.rounded_rectangle((dx - 2, y - 2, dx + font.getlength(draft_part) + 2, y + 27), radius=4, fill="#483b27")
                draw.text((dx, y), draft_part, font=font, fill=self.GOLD)
            cursor_x, cursor_y = x + font.getlength(visible.rstrip()), y
        if count < self.counts[key]:
            draw.rounded_rectangle((cursor_x + 3, cursor_y + 4, cursor_x + 5, cursor_y + 24), radius=1,
                                   fill=self.GOLD if pending else color)

    def frame(self, playback_t):
        cfg = self.config
        t = max(0, (playback_t - cfg.intro_seconds) * cfg.playback_rate)
        image = self.base.copy()
        draw = ImageDraw.Draw(image)
        rate_label = ("1× · реальное время" if cfg.playback_rate == 1 else
                      f"{cfg.playback_rate:g}× · одинаковое " +
                      ("замедление" if cfg.playback_rate < 1 else "ускорение"))
        self._text(draw, (1240, 47), rate_label, 16, self.MUTED, anchor="ra")
        self._text(draw, (1240, 82), f"{min(t, self.end):05.2f} с", 29, style="mono", anchor="ra")
        counts = {}
        for key, x, color in [("baseline", 40, self.BLUE), ("mtp", 654, self.TEAL)]:
            count, last, previous, pending = self._state(key, t)
            counts[key] = count
            self._draw_story(draw, x + 28, count, last, previous, pending, t, color, key)
            limit = self.counts[key]
            if count >= limit:
                status = f"ГОТОВО  ·  {self.ends[key]:.2f} с"
            elif pending:
                status = f"ПРОВЕРКА  ·  {len(pending['token_ids'])} draft-токена"
            elif not count:
                status = "PREFILL  ·  читаем промпт"
            elif key == "mtp" and last and last.get("round") is not None:
                accepted = last.get("accepted_count", 0)
                proposed = last.get("draft_count", 0)
                status = f"ПРИНЯТО {accepted}/{proposed}  ·  " + ("исправлено target" if accepted < proposed else "следующий блок")
            else:
                status = "ГЕНЕРАЦИЯ  ·  следующий токен"
            self._text(draw, (x + 28, 748), status, 13, self.GOLD if pending else color, "bold")
            self._text(draw, (x + 558, 746), f"{count} / {limit}", 16, style="mono", anchor="ra")
            draw.rounded_rectangle((x + 28, 778, x + 558, 783), radius=2, fill=self.TRACK)
            if count:
                draw.rounded_rectangle((x + 28, 778, x + 28 + 530 * count / limit, 783), radius=2, fill=color)
        done = all(n == self.counts[key] for key, n in counts.items())
        draw.rounded_rectangle((40, 809, 1240, 863), radius=13, fill="#17352f" if done else "#172132")
        if done:
            ratio = self.ends["baseline"] / self.ends["mtp"]
            saved = self.ends["baseline"] - self.ends["mtp"]
            if self.equal:
                self._text(draw, (62, 825), f"ТОЧНОЕ СОВПАДЕНИЕ  ·  {self.count} / {self.count} token IDs", 18, self.TEAL, "bold")
            else:
                self._text(draw, (62, 825), "ВЫВОДЫ РАЗЛИЧАЮТСЯ · показаны реальные токены", 17, self.RED, "bold")
            difference = f"{abs(saved):.2f} с " + ("раньше" if saved >= 0 else "позже")
            label = (f"{ratio:.2f}× до конца текста  ·  {difference}" if self.equal
                     else f"{self.ends['baseline']:.2f} с / {self.ends['mtp']:.2f} с")
            self._text(draw, (1218, 825), label, 18, anchor="ra")
        else:
            draw.ellipse((62, 830, 71, 839), fill=self.GOLD)
            self._text(draw, (82, 825), "предложено draft", 16, self.MUTED)
            draw.ellipse((296, 830, 305, 839), fill=self.TEAL)
            self._text(draw, (316, 825), "подтверждено target", 16, self.MUTED)
            lead = counts["mtp"] - counts["baseline"]
            label = f"MTP впереди на {lead} токенов" if lead >= 0 else f"Baseline впереди на {-lead} токенов"
            self._text(draw, (1218, 825), label, 18, anchor="ra")
        return image

    def render(self, output: Path):
        cfg = self.config
        seconds = cfg.intro_seconds + self.end / cfg.playback_rate + cfg.outro_seconds
        frame_count = math.ceil(seconds * cfg.fps)
        # One shared palette eliminates hue flicker from independent quantization.
        palette_source = self.frame(seconds)
        palette = palette_source.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
        output.parent.mkdir(parents=True, exist_ok=True)
        frames = (self.frame(i / cfg.fps).quantize(palette=palette, dither=Image.Dither.NONE)
                  for i in range(frame_count))
        StreamingGifWriter(output, round(1000 / cfg.fps)).write(frames)
        for fraction, name in [(0.36, "preview"), (1, "final")]:
            moment = seconds - 0.1 if name == "final" else cfg.intro_seconds + self.end / cfg.playback_rate * fraction
            self.frame(moment).save(output.with_name(output.stem + f"-{name}.png"))
        return {"file": str(output.resolve()), "playback_rate": cfg.playback_rate,
                "fps": cfg.fps, "duration_seconds": seconds, "frames": frame_count,
                "encoded_duration_seconds": frame_count / cfg.fps,
                "frame_interval_seconds": 1 / cfg.fps,
                "visible_output_tokens": self.count,
                "stopped_before_first_eos": self.stop_reasons["baseline"] == "first_eos",
                "display_stop_reason_per_lane": self.stop_reasons,
                "first_eos_index_per_lane": self.eos_positions,
                "request_seconds_to_visible_completion": self.ends,
                "speedup_request": self.ends["baseline"] / self.ends["mtp"] if self.equal else None,
                "all_captured_token_ids_equal": self.equal,
                "visible_output_tokens_per_lane": self.counts, "scrolling_text": self.scrolling,
                "timing_policy": "each lane uses its own observed request-relative timestamps; same playback rate, no artificial lead",
                "prompt_display": "actual prompt, wrapped to at most three lines; exact full prompt stored in trace.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "artifacts/demos/mtp-race/trace.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/demos/mtp-race")
    parser.add_argument("--rates", type=float, nargs="+", default=[1.0, 0.25])
    parser.add_argument("--fps", type=int)
    parser.add_argument("--max-visible-tokens", type=int)
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    reports = []
    for rate in args.rates:
        if not math.isfinite(rate) or not 0 < rate <= 32:
            parser.error("Playback rates must be finite and in (0, 32]")
        name = "mtp-real-time.gif" if rate == 1 else f"mtp-{rate:g}x.gif"
        replay = GenerationReplay(trace, ReplayConfig(playback_rate=rate, fps=args.fps or (50 if rate >= 1 else 25),
                                                   max_visible_tokens=args.max_visible_tokens, allow_mismatch=args.allow_mismatch))
        report = replay.render(args.output / name)
        report["trace_sha256"] = sha256(args.trace.read_bytes()).hexdigest()
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    (args.output / "render-manifest.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
