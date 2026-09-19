"""Build a replay for one dataset trajectory from saved benchmark events."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from inference_lab.core.config import ROOT
from inference_lab.core.io import sha256_file, write_json
from inference_lab.benchmarking.speculative_report import SpeculativeReport
from .recording import validate_generation_trace
from .trace import display_tokens, _enrich_events, parity


class BenchmarkReplayBuilder:
    """Use only recorded events; tokenization here adds display text, not timings."""

    def __init__(self, baseline: Path, mtp: Path, index: int, tokenizer=None):
        if type(index) is not int or index < 0:
            raise ValueError("Dataset index must be a nonnegative integer")
        self.baseline, self.mtp = Path(baseline).resolve(), Path(mtp).resolve()
        self.index, self.tokenizer = index, tokenizer
        self.series_evidence = None

    RAW_FILES = ("summary.json", "prompts.json", "samples.jsonl")
    TOKENIZER_FILES = frozenset({
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model",
        "spiece.model", "sentencepiece.bpe.model", "chat_template.jinja",
    })

    @classmethod
    def from_series(cls, directory: Path, index: int, tokenizer=None):
        path = Path(directory)
        if path.is_dir():
            path = path / "manifest.json"
        path = path.resolve()
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "completed":
            raise ValueError("The series must be completed; use explicit completed --baseline/--mtp blocks while it is running")
        merged = manifest.get("merged_runs") or {}
        if not merged.get("baseline") or not merged.get("mtp"):
            raise ValueError("The series has no merged results yet; use explicit completed --baseline/--mtp run directories")

        def resolve(value):
            candidate = Path(value)
            return (candidate if candidate.is_absolute() else path.parent / candidate).resolve()

        builder = cls(resolve(merged["baseline"]), resolve(merged["mtp"]), index, tokenizer)
        files = manifest.get("merged_files")
        if not isinstance(files, dict) or set(files) != {"baseline", "mtp"}:
            raise ValueError("Completed series is missing merged_files hash evidence")
        evidence = deepcopy(files)
        for role, entries in evidence.items():
            if not isinstance(entries, dict) or set(entries) != set(cls.RAW_FILES):
                raise ValueError(f"Series has incomplete merged-file evidence for {role}")
            for name, entry in entries.items():
                if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))):
                    raise ValueError(f"Series has invalid merged-file evidence for {role}/{name}")
                entry["path"] = str(resolve(entry["path"]))
        builder.series_evidence = {"manifest_path": str(path), "manifest_sha256": sha256_file(path),
                                   "merged_files": evidence}
        builder._verify_series_evidence()
        return builder

    def _verify_series_evidence(self):
        if self.series_evidence is None:
            return
        manifest_path = Path(self.series_evidence["manifest_path"])
        if sha256_file(manifest_path) != self.series_evidence["manifest_sha256"]:
            raise ValueError("Series manifest changed after replay selection")
        for role, directory in (("baseline", self.baseline), ("mtp", self.mtp)):
            for name, entry in self.series_evidence["merged_files"][role].items():
                path = directory / name
                if Path(entry["path"]).resolve() != path.resolve() or sha256_file(path) != entry["sha256"]:
                    raise ValueError(f"Series merged-file hash/path mismatch: {role}/{name}")

    @classmethod
    def _verified_tokenizer_assets(cls, directory, expected_manifest_hash):
        manifest_path = directory / "download-manifest.json"
        raw = manifest_path.read_bytes()
        checksum = sha256(raw).hexdigest()
        if checksum != expected_manifest_hash:
            raise ValueError("Tokenizer model manifest differs from the benchmark's pinned manifest")
        manifest = json.loads(raw)
        records = manifest.get("files")
        if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
            raise ValueError("Tokenizer model manifest has invalid file evidence")
        names = [item.get("path") for item in records]
        if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValueError("Tokenizer model manifest has invalid or duplicate file paths")
        indexed = {item["path"]: item for item in records}
        if not {"tokenizer.json", "tokenizer_config.json"}.issubset(indexed):
            raise ValueError("Pinned model manifest must include tokenizer.json and tokenizer_config.json")
        # Only these small tokenizer/config assets are opened. Never scan model weights.
        selected = cls.TOKENIZER_FILES.intersection(indexed)
        unexpected = [name for name in cls.TOKENIZER_FILES - selected if (directory / name).exists()]
        if unexpected:
            raise ValueError(f"Unpinned tokenizer assets could override decoding: {sorted(unexpected)}")
        assets = {}
        for name in sorted(selected):
            path, expected = directory / name, indexed[name]
            if (not path.is_file() or path.stat().st_size != expected.get("bytes")
                    or sha256_file(path) != expected.get("sha256")):
                raise ValueError(f"Pinned tokenizer asset hash/size mismatch: {name}")
            assets[name] = {"path": str(path), "bytes": expected["bytes"], "sha256": expected["sha256"]}
        return {"status": "verified", "model_directory": str(directory),
                "manifest": {"path": str(manifest_path), "sha256": checksum,
                             "repo_id": manifest.get("repo_id"), "resolved_revision": manifest.get("resolved_revision")},
                "assets": assets, "verification_scope": "Pinned tokenizer/config assets only; no model weight files read"}

    def _load_tokenizer(self, summary):
        recorded = Path(summary["config"]["model_path"]).expanduser().resolve()
        selected = recorded if recorded.is_dir() else ROOT / "models/qwen3.5-9b-mlx-4bit"
        provenance = self._verified_tokenizer_assets(selected, summary["model_manifest_sha256"])
        provenance.update(recorded_model_directory=str(recorded),
                          selection="recorded_model_directory" if selected == recorded else "verified_project_fallback",
                          loader="transformers.AutoTokenizer.from_pretrained; local_files_only=True; trust_remote_code=False")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(selected), local_files_only=True, trust_remote_code=False)
        provenance["tokenizer_class"] = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        return tokenizer, provenance

    @staticmethod
    def _selected(directory: Path, index: int):
        summary = json.loads((directory / "summary.json").read_text())
        prompt = next((p for p in json.loads((directory / "prompts.json").read_text()) if p["index"] == index), None)
        measurement = None
        with (directory / "samples.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["index"] == index:
                    measurement = row
                    break
        if prompt is None or measurement is None:
            raise ValueError(f"Dataset index {index} is missing from {directory.name}")
        return summary, prompt, measurement

    def build(self):
        self._verify_series_evidence()
        report = SpeculativeReport(self.baseline, [self.mtp]).analyze()
        if report["status"] != "completed":
            raise ValueError(f"Incomparable benchmark logs: {report['errors']}")
        lanes, metadata, prompt_ids, prompt_text, eos = {}, {}, None, None, None
        display_prompt = None
        tokenizer = self.tokenizer
        tokenizer_provenance = (None if tokenizer is None else
                                {"status": "injected_unverified", "selection": "caller_supplied_tokenizer",
                                 "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                                 "verification_scope": "Explicitly injected tokenizer; no asset provenance is asserted"})
        for key, directory in (("baseline", self.baseline), ("mtp", self.mtp)):
            summary, prompt, row = self._selected(directory, self.index)
            if summary["config"].get("trace_generation") is not True:
                raise ValueError("This run has no recorded events; aggregate timings cannot reconstruct a real trajectory")
            validate_generation_trace(row)
            if prompt_ids is not None and prompt_ids != prompt["prompt_tokens"]:
                raise ValueError("Selected inputs differ")
            prompt_ids = prompt["prompt_tokens"]
            if tokenizer is None:
                tokenizer, tokenizer_provenance = self._load_tokenizer(summary)
            if prompt_text is None:
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
                display_prompt = prompt.get("problem") or prompt_text
            lane = deepcopy(row["generation_trace"])
            if eos is not None and lane["eos_token_ids"] != eos:
                raise ValueError("EOS definitions differ between saved trajectories")
            eos = lane["eos_token_ids"]
            lane.update(display_tokens(tokenizer, lane["token_ids"]))
            _enrich_events(lane, tokenizer)
            lanes[key] = lane
            metadata[key] = {
                "source_run": directory.name,
                "source_files": {name: {"sha256": sha256((directory / name).read_bytes()).hexdigest()}
                                 for name in ("summary.json", "prompts.json", "samples.jsonl")},
                "original_problem": prompt.get("problem"),
                "instrumentation": lane["instrumentation"],
            }
        self._verify_series_evidence()
        return {
            "schema_version": 1, "prompt": display_prompt, "prompt_token_ids": prompt_ids,
            "max_new_tokens": len(lanes["baseline"]["token_ids"]), "eos_token_ids": eos,
            "metadata": {"dataset_index": self.index, "enable_thinking": True,
                         "tokenizer": tokenizer_provenance, "series": self.series_evidence,
                         "decoded_input_text": prompt_text,
                         "display_prompt_source": "saved original problem; complete formatted input retained as token IDs and decoded_input_text",
                         "sampling": "greedy", "ignore_eos": True,
                         "timing_origin": "each lane's actual backend prefill start; runs executed sequentially",
                         "source": "saved benchmark generation_trace; no inference or invented timestamps",
                         "display_policy": "offline token decoding; incomplete Unicode prefixes delayed; stop display before first EOS",
                         **metadata},
            **lanes, "parity": parity(lanes["baseline"]["token_ids"], lanes["mtp"]["token_ids"]),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, help="Series directory or manifest.json")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--mtp", type=Path)
    parser.add_argument("--index", type=int, required=True, help="Original zero-based dataset index, 0..99")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rates", type=float, nargs="+", default=[1.0])
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-visible-tokens", type=int, help="Optional prefix preview; full log stays unchanged")
    parser.add_argument("--trace-only", action="store_true", help="Create replay trace without encoding a GIF")
    args = parser.parse_args(argv)
    if args.series and (args.baseline or args.mtp):
        parser.error("Use --series OR the --baseline/--mtp pair")
    if not args.series and not (args.baseline and args.mtp):
        parser.error("Provide --series or both --baseline and --mtp")
    if any(not math.isfinite(rate) or not 0 < rate <= 32 for rate in args.rates):
        parser.error("Playback rates must be finite and in (0, 32]")
    builder = (BenchmarkReplayBuilder.from_series(args.series, args.index) if args.series else
               BenchmarkReplayBuilder(args.baseline, args.mtp, args.index))
    trace = builder.build()
    output = args.output or ROOT / "artifacts/demos" / f"sample-{args.index:03d}"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "trace.json"
    write_json(target, trace)
    print(f"Replay trace: {target}", flush=True)
    if not trace["parity"]["equal"]:
        print("Output token IDs differ; the replay will show both actual trajectories with a mismatch banner.", flush=True)
    if args.trace_only:
        return 0
    from .render import GenerationReplay, ReplayConfig
    reports = []
    for rate in args.rates:
        config = ReplayConfig(playback_rate=rate, fps=args.fps,
                              max_visible_tokens=args.max_visible_tokens, allow_mismatch=True)
        name = "mtp-real-time.gif" if rate == 1 else f"mtp-{rate:g}x.gif"
        result = GenerationReplay(trace, config).render(output / name)
        result["trace_sha256"] = sha256(target.read_bytes()).hexdigest()
        result["dataset_index"] = args.index
        reports.append(result)
        print(f"GIF: {output / name}", flush=True)
    write_json(output / "render-manifest.json", reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
