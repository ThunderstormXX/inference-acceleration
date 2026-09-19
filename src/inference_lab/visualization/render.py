"""A typography-led GIF replay of observed, unmodified inference event times."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont

from inference_lab.core.config import ROOT


@dataclass(frozen=True)
class ReplayConfig:
    width: int = 1280
    height: int = 920
    fps: int = 25
    playback_rate: float = 1.0
    intro_seconds: float = 1.4
    outro_seconds: float = 2.8


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
        if not trace.get("parity", {}).get("equal"):
            raise ValueError("A matching-output replay requires verified exact token parity")
        self.trace, self.config = trace, config
        self.type = Typography()
        self.ids = trace["baseline"]["token_ids"]
        if self.ids != trace["mtp"]["token_ids"]:
            raise ValueError("Trace parity flag disagrees with actual token IDs")
        for key in ("baseline", "mtp"):
            lane = trace[key]
            if len(lane["token_texts"]) != len(self.ids):
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
            if committed != self.ids:
                raise ValueError("Commit events disagree with final generated token IDs")
        eos = set(trace.get("eos_token_ids", trace.get("metadata", {}).get("eos_token_ids", [])))
        self.count = next((i for i, token in enumerate(self.ids) if token in eos), len(self.ids))
        if self.count < 2:
            raise ValueError("Replay requires at least two visible output tokens")
        texts = trace["baseline"]["token_texts"][:self.count]
        self.prefixes = [""]
        for text in texts:
            self.prefixes.append(self.prefixes[-1] + text)
        self.full_text = self.prefixes[-1]
        self.ends = {}
        for key in ("baseline", "mtp"):
            commits = [e for e in trace[key]["events"] if e["type"] == "commit"]
            if any(a["t"] > b["t"] for a, b in zip(commits, commits[1:])):
                raise ValueError("Commit events must be chronological")
            self.ends[key] = next(e["t"] for e in commits if e["output_count"] >= self.count)
        self.end = max(self.ends.values())
        self.font_size = 23
        for size in (23, 22, 21, 20, 19):
            self.font_size = size
            if len(self._lines(self.full_text, 522)) <= 12:
                break
        if len(self._lines(self.full_text, 522)) > 12:
            raise ValueError("Output too long for the replay: choose a shorter complete story")
        self.base = self._base()

    def _lines(self, text: str, width: int):
        font = self.type(self.font_size, "serif")
        lines, start = [], 0
        # Preserve exact whitespace/character indices, including token boundaries
        # inside words; wrapping itself never changes the displayed token stream.
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
                line += part
            lines.append((start, line + ending))
            start += len(line + ending)
        if not lines:
            lines.append((0, ""))
        return lines

    def _text(self, draw, xy, text, size=18, color=None, style="regular", anchor=None):
        draw.text(xy, text, fill=color or self.INK, font=self.type(size, style), anchor=anchor)

    def _base(self):
        image = Image.new("RGB", (self.config.width, self.config.height), self.BG)
        draw = ImageDraw.Draw(image)
        # Restrained background light, fixed across frames to preserve GIF quality.
        self._text(draw, (40, 30), "QWEN3.5 / 9B / APPLE SILICON", 14, self.MUTED, "mono")
        self._text(draw, (40, 57), "Один текст. Два ритма.", 42, style="bold")
        subtitle = (f"Начало рассуждения · первые {self.count} токенов · native MTP"
                    if self.trace.get("metadata", {}).get("enable_thinking")
                    else "Обычная генерация и speculative decoding с native MTP")
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
        for x, title, sub, color in [
            (40, "Обычный decode", "1 токен за шаг большой модели", self.BLUE),
            (654, "MTP · блок 3", "2 draft-токена → проверка target", self.TEAL),
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
        count = min(commits[-1]["output_count"], self.count) if commits else 0
        last = commits[-1] if commits else None
        previous = min(commits[-2]["output_count"], self.count) if len(commits) > 1 else 0
        pending = None
        if key == "mtp" and count < self.count:
            proposals = [e for e in events if e["type"] == "draft" and e["t"] <= t]
            if proposals:
                candidate = proposals[-1]
                if last is None or candidate["t"] > last["t"]:
                    pending = candidate
        return count, last, previous, pending

    def _draw_story(self, draw, x, count, last, previous, pending, t, color):
        committed = self.prefixes[count]
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
        layout = self._lines(self.full_text if self.full_text.startswith(text) else text, 522)
        cursor_x, cursor_y = x, 380
        fresh = last is not None and 0 <= t - last["t"] < 0.18
        for row, (start, line) in enumerate(layout[:12]):
            y = 380 + row * line_height
            visible = text[start:start + len(line)].rstrip("\r\n")
            if not visible:
                if start >= len(text):
                    break
                continue
            if fresh:
                lo = max(0, len(self.prefixes[previous]) - start)
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
        if count < self.count:
            draw.rounded_rectangle((cursor_x + 3, cursor_y + 4, cursor_x + 5, cursor_y + 24), radius=1,
                                   fill=self.GOLD if pending else color)

    def frame(self, playback_t):
        cfg = self.config
        t = max(0, (playback_t - cfg.intro_seconds) * cfg.playback_rate)
        image = self.base.copy()
        draw = ImageDraw.Draw(image)
        rate_label = "1× · реальное время" if cfg.playback_rate == 1 else f"{cfg.playback_rate:g}× · одинаковое замедление"
        self._text(draw, (1240, 47), rate_label, 16, self.MUTED, anchor="ra")
        self._text(draw, (1240, 82), f"{min(t, self.end):05.2f} с", 29, style="mono", anchor="ra")
        counts = {}
        for key, x, color in [("baseline", 40, self.BLUE), ("mtp", 654, self.TEAL)]:
            count, last, previous, pending = self._state(key, t)
            counts[key] = count
            self._draw_story(draw, x + 28, count, last, previous, pending, t, color)
            if count >= self.count:
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
            self._text(draw, (x + 558, 746), f"{count} / {self.count}", 16, style="mono", anchor="ra")
            draw.rounded_rectangle((x + 28, 778, x + 558, 783), radius=2, fill=self.TRACK)
            if count:
                draw.rounded_rectangle((x + 28, 778, x + 28 + 530 * count / self.count, 783), radius=2, fill=color)
        done = all(n == self.count for n in counts.values())
        draw.rounded_rectangle((40, 809, 1240, 863), radius=13, fill="#17352f" if done else "#172132")
        if done:
            ratio = self.ends["baseline"] / self.ends["mtp"]
            saved = self.ends["baseline"] - self.ends["mtp"]
            self._text(draw, (62, 825), f"ТОЧНОЕ СОВПАДЕНИЕ  ·  {self.count} / {self.count} token IDs", 18, self.TEAL, "bold")
            difference = f"{abs(saved):.2f} с " + ("раньше" if saved >= 0 else "позже")
            self._text(draw, (1218, 825), f"{ratio:.2f}× до конца текста  ·  {difference}", 18, anchor="ra")
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
        frames = []
        for i in range(frame_count):
            frame = self.frame(i / cfg.fps)
            frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=round(1000 / cfg.fps), loop=0, optimize=True, disposal=1)
        for fraction, name in [(0.36, "preview"), (1, "final")]:
            moment = seconds - 0.1 if name == "final" else cfg.intro_seconds + self.end / cfg.playback_rate * fraction
            self.frame(moment).save(output.with_name(output.stem + f"-{name}.png"))
        return {"file": str(output.resolve()), "playback_rate": cfg.playback_rate,
                "fps": cfg.fps, "duration_seconds": seconds, "frames": frame_count,
                "visible_output_tokens": self.count, "stopped_before_first_eos": self.count < len(self.ids),
                "request_seconds_to_visible_completion": self.ends,
                "speedup_request": self.ends["baseline"] / self.ends["mtp"],
                "all_captured_token_ids_equal": True,
                "timing_policy": "each lane uses its own observed request-relative timestamps; same playback rate, no artificial lead",
                "prompt_display": "actual prompt, wrapped to at most three lines; exact full prompt stored in trace.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "artifacts/demos/mtp-race/trace.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/demos/mtp-race")
    parser.add_argument("--rates", type=float, nargs="+", default=[1.0, 0.25])
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    reports = []
    for rate in args.rates:
        if not 0 < rate <= 2:
            parser.error("Playback rates must be in (0, 2]")
        name = "mtp-real-time.gif" if rate == 1 else f"mtp-{rate:g}x.gif"
        replay = GenerationReplay(trace, ReplayConfig(playback_rate=rate, fps=50 if rate >= 1 else 25))
        report = replay.render(args.output / name)
        report["trace_sha256"] = sha256(args.trace.read_bytes()).hexdigest()
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    (args.output / "render-manifest.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
