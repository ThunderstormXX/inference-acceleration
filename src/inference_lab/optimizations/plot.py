"""Static, protocol-separated figures from completed benchmark summaries."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tempfile

from .report import read_experiment

DFLASH = (
    ("ar", "AR baseline"),
    ("dflash", "DFlash · stock"),
    ("dflash-target", "DFlash · target projections"),
    ("dflash-target-argmax", "DFlash · target + argmax"),
    ("dflash-all-argmax", "DFlash · all + argmax"),
)
MTP = (("ar_k1", "AR baseline"), ("mtp_k3", "MTP · K=3"))
VOCAB = (("ar_k1", "AR baseline"), ("mtp-stock_k3", "MTP · stock K=3"),
         ("mtp_k3", "MTP · shortlist K=3"))


def load_panel(path: Path, keys, *, section: str) -> dict:
    directory = path if path.is_dir() else path.parent
    if not path.is_dir() and path.name != "summary.json":
        raise ValueError("Pass a benchmark directory or its summary.json")
    experiment = read_experiment(directory)
    expected_family = "kernel_sweep" if section == "by_variant" else "mtp_sweep"
    if experiment["family"] != expected_family:
        raise ValueError(f"Expected {expected_family} input: {directory}; "
                         + "; ".join(experiment["warnings"]))
    validated = {row["name"]: row for row in experiment["rows"]}
    rows = []
    for key, label in keys:
        entry = validated.get(key)
        if entry is None or entry.get("eligible_for_same_protocol_comparison") is not True:
            reasons = [*experiment["warnings"]]
            if entry is None:
                reasons.append("requested variant is missing")
            else:
                reasons.extend(entry["no_claim_reasons"])
                reasons.extend(entry["warnings"])
            raise ValueError(f"Refusing diagnostic plot input {directory.name}/{key}: "
                             + "; ".join(dict.fromkeys(reasons)))
        record = entry["reported_summary"]
        stat = record.get("tok_s", {})
        n, mean, sd = stat.get("n"), stat.get("mean"), stat.get("stdev")
        if (type(n) is not int or n < 2 or type(mean) not in (int, float)
                or type(sd) not in (int, float) or not math.isfinite(mean)
                or not math.isfinite(sd) or mean <= 0 or sd < 0):
            raise ValueError(f"Need valid mean/sample SD from at least two runs: {key}")
        # The shared reporter checks counts and means; do not draw a stale SD either.
        raw_sd = entry["tok_s"].get("stdev")
        if raw_sd is None or not math.isclose(sd, raw_sd, rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError(f"Summary/raw sample SD differs: {directory.name}/{key}")
        rows.append({"key": key, "label": label, "mean": mean, "sd": sd, "runs": n})
    return {"rows": rows, "protocol": experiment["protocol"], "source": directory.name}


def make_plot(dflash: Path, mtp: Path, output: Path, vocab: Path | None = None) -> tuple[Path, Path]:
    panels = [load_panel(dflash, DFLASH, section="by_variant"), load_panel(mtp, MTP, section="by_mode")]
    if vocab is not None:
        panels.append(load_panel(vocab, VOCAB, section="by_mode"))
        if panels[-1]["protocol"].get("compare_stock_draft") is not True:
            raise ValueError("Vocabulary panel requires direct stock-draft comparison protocol")
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="inference-lab-mpl-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#C4CCD5",
                         "text.color": "#172A3A", "axes.labelcolor": "#405366",
                         "xtick.color": "#596B7D", "ytick.color": "#172A3A",
                         "svg.fonttype": "none"})
    wide = len(panels) == 3
    figure, axes = plt.subplots(1, len(panels), figsize=(22 if wide else 14.4, 6.8),
                                gridspec_kw={"width_ratios": [1.28, 1, 1.12] if wide else [1.28, 1]})
    figure.subplots_adjust(left=.135 if wide else .205, right=.97, top=.72, bottom=.27,
                           wspace=.77 if wide else .68)
    figure.suptitle("M2 Pro · Qwen3.5-9B · 4-bit", x=.05, y=.965, ha="left", fontsize=22, weight="bold")
    figure.text(.05, .89, "Separate benchmark sessions and protocols — compare variants within each panel",
                fontsize=12, color="#405366")
    for index, (axis, panel) in enumerate(zip(axes, panels)):
        rows, protocol = panel["rows"], panel["protocol"]
        title = ("DFlash kernel ablations", "Native MTP confirmation", "Draft vocabulary ablation")[index]
        prompts = len(protocol["prompts"]) if isinstance(protocol.get("prompts"), list) else "?"
        subtitle = f"{prompts} prompts · {protocol.get('tokens', '?')} output tokens · {protocol.get('repeats', '?')} repeats"
        axis.set_title(title + "\n" + subtitle, loc="left", pad=24, fontsize=12, linespacing=1.7, weight="bold")
        positions = list(range(len(rows)))
        colors = ["#8393A3" if row["key"] in {"ar", "ar_k1"} else "#277DA8" for row in rows]
        axis.barh(positions, [row["mean"] for row in rows], height=.56, color=colors,
                  xerr=[row["sd"] for row in rows],
                  error_kw={"ecolor": "#172A3A", "capsize": 4, "elinewidth": 1.4}, zorder=3)
        axis.set_yticks(positions, [row["label"] for row in rows])
        axis.tick_params(axis="y", length=0, pad=10)
        axis.invert_yaxis()
        high = max(row["mean"] + row["sd"] for row in rows)
        axis.set_xlim(0, high * 1.35)
        axis.set_ylim(len(rows) - .35, -.65)
        axis.xaxis.grid(True, color="#E5EAF0", linewidth=.8, zorder=0)
        axis.set_xlabel("Decode throughput (tokens / second)", labelpad=13)
        for position, row in zip(positions, rows):
            axis.text(row["mean"] + row["sd"] + high * .035, position,
                      f"{row['mean']:.2f} ± {row['sd']:.2f}", va="center", fontsize=10, weight="normal")
        counts = sorted({row["runs"] for row in rows})
        if len(counts) == 1:
            count_text = f"n = {counts[0]} requests per variant; {prompts} unique prompts"
        elif index == 2 and rows[1]["runs"] == rows[2]["runs"]:
            count_text = f"n = {rows[0]['runs']} AR; {rows[1]['runs']} per MTP; {prompts} unique prompts"
        else:
            count_text = "Request counts: " + ", ".join(f"{r['key']}={r['runs']}" for r in rows)
        axis.text(0, -.22, count_text, transform=axis.transAxes, fontsize=9, color="#596B7D")
    figure.text(.05, .105, "Bars: request mean. Error bars: sample SD, not a confidence interval.", fontsize=10)
    figure.text(.05, .063, "Repeated prompts are dependent. Absolute rates across panels are not a matched comparison. "
                "First output token belongs to prefill.", fontsize=9, color="#596B7D")
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("") if output.suffix.lower() in {".png", ".svg"} else output
    png, svg = Path(str(stem) + ".png"), Path(str(stem) + ".svg")
    figure.savefig(png, dpi=180, facecolor="white")
    figure.savefig(svg, facecolor="white", metadata={"Date": None})
    plt.close(figure)
    return png, svg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dflash", type=Path, required=True, help="Completed kernel benchmark directory or summary.json")
    parser.add_argument("--mtp", type=Path, required=True, help="Completed MTP benchmark directory or summary.json")
    parser.add_argument("--vocab", type=Path, help="Optional completed stock-versus-shortlist MTP comparison")
    parser.add_argument("--output", type=Path, required=True, help="PNG/SVG path or stem; both formats are saved")
    args = parser.parse_args(argv)
    for path in make_plot(args.dflash, args.mtp, args.output, args.vocab):
        print(path)
    return 0
