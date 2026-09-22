"""Reconstruct one exact neighboring AR/MTP sweep pair without inference."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import re

from ..benchmarking.metrics import validate_measurement
from ..core.config import ROOT
from ..core.io import write_json
from .recording import validate_generation_trace
from .replay_logs import BenchmarkReplayBuilder
from .trace import display_tokens, _enrich_events, parity

MAX_RAW_BYTES = 128 * 1024 * 1024


def _read_bounded(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_RAW_BYTES + 1)
    if len(payload) > MAX_RAW_BYTES:
        raise ValueError("Replay source exceeds the 128 MiB offline limit")
    return payload


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Missing or invalid SHA256: {name}")
    return value


def _verified_source(path, expected, fallback=None):
    _digest(expected, str(path))
    recorded = Path(path).expanduser().resolve()
    selected = recorded if recorded.is_file() else fallback
    if selected is None or not selected.is_file():
        raise ValueError(f"Missing pinned replay source: {recorded}")
    selected = selected.resolve()
    payload = _read_bounded(selected)
    if sha256(payload).hexdigest() != expected:
        raise ValueError(f"Replay source hash mismatch: {selected}")
    return payload, {"path": str(selected), "recorded_path": str(recorded), "sha256": expected}


class SweepReplayBuilder:
    """Fail closed on incomplete logs, ambiguous pairs and unpinned assets."""

    def __init__(self, input_path, prompt_index=0, repeat=0, block_size=3):
        for name, value, low, high in (("prompt_index", prompt_index, 0, 4),
                                       ("repeat", repeat, 0, 4), ("block_size", block_size, 2, 5)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
        self.input = Path(input_path).expanduser().resolve()
        self.prompt_index, self.repeat, self.block_size = prompt_index, repeat, block_size
        self.payload = _read_bounded(self.input)
        self.input_sha256 = sha256(self.payload).hexdigest()
        if self.input.name.endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(self.payload)) as stream:
                document = stream.read(MAX_RAW_BYTES + 1)
            if len(document) > MAX_RAW_BYTES:
                raise ValueError("Decompressed replay source exceeds the 128 MiB offline limit")
        else:
            document = self.payload
        self.document_sha256 = sha256(document).hexdigest()
        self.report = json.loads(document)

    def _selection(self):
        report = self.report
        if report.get("schema_version") != 1 or report.get("status") != "completed":
            raise ValueError("Replay requires a completed schema-version-1 sweep")
        protocol = report.get("protocol", {})
        if protocol.get("trace") is not True:
            raise ValueError("Sweep has no recorded trace; timings cannot reconstruct real events")
        prompts, runs = protocol.get("prompts"), report.get("runs")
        if (not isinstance(prompts, list) or self.prompt_index >= len(prompts)
                or not isinstance(runs, list)):
            raise ValueError("Missing selected prompt or sweep runs")
        if type(protocol.get("expected_runs")) is not int or len(runs) != protocol["expected_runs"]:
            raise ValueError("Completed sweep run count differs from its protocol")
        selected = [(index, row) for index, row in enumerate(runs)
                    if row.get("mode") == "mtp" and row.get("role") == "paired"
                    and (row.get("prompt_index"), row.get("repeat"), row.get("block_size"))
                    == (self.prompt_index, self.repeat, self.block_size)]
        if len(selected) != 1:
            raise ValueError("Need exactly one selected treatment mode=mtp; stock-draft rows are not treatments")
        mtp_index, mtp = selected[0]
        pair_id = mtp.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("Selected MTP row has no pair ID")
        pair = [(index, row) for index, row in enumerate(runs) if row.get("pair_id") == pair_id]
        if len(pair) != 2 or sorted(row.get("mode", "") for _, row in pair) != ["ar", "mtp"]:
            raise ValueError("Pair ID must identify exactly one AR and one treatment MTP")
        ar_index, ar = next((index, row) for index, row in pair if row["mode"] == "ar")
        if abs(ar_index - mtp_index) != 1:
            raise ValueError("Selected AR/MTP pair is not neighboring in the measured run order")
        for row in (ar, mtp):
            if (row.get("role") != "paired" or row.get("prompt_index") != self.prompt_index
                    or row.get("repeat") != self.repeat or row.get("paired_block_size") != self.block_size):
                raise ValueError("AR/MTP pair metadata differs")
            if row.get("sleep", {}).get("sleep_detected") is not False:
                raise ValueError("Selected measurement lacks an awake-host observation")
        if ar.get("block_size") != 1 or self.block_size not in protocol.get("block_sizes", []):
            raise ValueError("Pair block sizes differ from the protocol")
        prompt = prompts[self.prompt_index]
        ids = prompt.get("prompt_tokens")
        if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError("Selected prompt has invalid token IDs")
        for row in (ar, mtp):
            measurement = row["measurement"]
            validate_measurement(measurement)
            validate_generation_trace(measurement)
            if (measurement["prompt_tokens"] != len(ids)
                    or measurement["generated_tokens"] != protocol.get("tokens")):
                raise ValueError("Selected measurement differs from the fixed prompt/output token budget")
        comparison = parity(ar["measurement"]["generated_token_ids"], mtp["measurement"]["generated_token_ids"])
        if not comparison["equal"]:
            raise ValueError("Selected AR/MTP output IDs differ; exact replay requires matching full outputs")
        return prompt, ar, mtp, ar_index, mtp_index, comparison

    def _prompt_provenance(self):
        protocol = self.report["protocol"]
        recorded = protocol.get("prompts_source")
        if not isinstance(recorded, str) or not recorded:
            raise ValueError("Missing pinned prompt source")
        payload, evidence = _verified_source(recorded, protocol.get("prompts_sha256"),
                                             ROOT / "configs/profiling" / Path(recorded).name)
        original = json.loads(payload)
        if not isinstance(original, list) or original[:len(protocol["prompts"])] != protocol["prompts"]:
            raise ValueError("Embedded prompts differ from the pinned prompt source")
        return evidence

    def _tokenizer(self):
        report = self.report
        backend, baseline = report.get("backend", {}), report.get("baseline", {})
        if backend.get("framework") != "mlx-vlm-mtp" or baseline.get("framework") != "mlx-vlm":
            raise ValueError("Replay requires matched MLX-VLM AR/native-MTP implementations")
        paths = [value.get("model_path") for value in (backend, baseline)]
        if (any(not isinstance(path, str) or not path for path in paths)
                or Path(paths[0]).expanduser().resolve() != Path(paths[1]).expanduser().resolve()):
            raise ValueError("AR/MTP target model directories differ")
        for key in ("model_type", "weight_bits", "weight_group_size", "kv_bits", "versions"):
            if key not in backend or key not in baseline or backend[key] != baseline[key]:
                raise ValueError(f"AR/MTP target metadata differs: {key}")
        for key, expected in (("sampling", "greedy"), ("ignore_eos", True),
                              ("batch_size", 1), ("fresh_cache_per_request", True)):
            if backend.get(key) != expected or baseline.get(key) != expected:
                raise ValueError(f"Unsupported AR/MTP sampling/cache protocol: {key}")
        manifest_hashes = [value for value in (
            report.get("target_manifest_sha256"), report.get("model_manifest_sha256"),
            backend.get("target_manifest_sha256"), backend.get("model_manifest_sha256"),
            baseline.get("target_manifest_sha256"), baseline.get("model_manifest_sha256")) if value is not None]
        vocabulary_evidence = None
        vocabulary = report.get("draft_vocabulary")
        if vocabulary is not None:
            recorded = vocabulary.get("vocabulary_path")
            if not isinstance(recorded, str) or not recorded:
                raise ValueError("Missing pinned draft vocabulary path")
            payload, vocabulary_evidence = _verified_source(recorded, vocabulary.get("vocabulary_sha256"),
                ROOT / "configs/optimizations" / Path(recorded).name)
            config = json.loads(payload)
            if config.get("kind") != "mtp_draft_only_vocabulary":
                raise ValueError("Pinned artifact is not an MTP draft vocabulary")
            if (config.get("shortlist_size") != vocabulary.get("shortlist_size")
                    or config.get("target_vocab_size") != vocabulary.get("target_vocab_size")):
                raise ValueError("Draft vocabulary metadata differs from its pinned artifact")
            manifest_hashes.append(config.get("target_manifest_sha256"))
            vocabulary_evidence["tokenizer_sha256"] = _digest(config.get("tokenizer_sha256"), "vocabulary tokenizer")
        if not manifest_hashes:
            raise ValueError("Sweep has no recorded target manifest hash or pinned draft-vocabulary provenance")
        for value in manifest_hashes:
            _digest(value, "target manifest")
        if len(set(manifest_hashes)) != 1:
            raise ValueError("Recorded target manifest hashes disagree")
        summary = {"config": {"model_path": paths[0]}, "model_manifest_sha256": manifest_hashes[0]}
        # Reuse the established offline loader: only pinned tokenizer/config
        # assets are read, never model weights or remote custom code.
        loader = BenchmarkReplayBuilder(self.input.parent, self.input.parent, self.prompt_index)
        tokenizer, evidence = loader._load_tokenizer(summary)
        if (vocabulary_evidence is not None
                and evidence["assets"]["tokenizer.json"]["sha256"] != vocabulary_evidence["tokenizer_sha256"]):
            raise ValueError("Pinned draft vocabulary tokenizer differs from replay tokenizer")
        return tokenizer, evidence, vocabulary_evidence

    def build(self):
        if sha256(_read_bounded(self.input)).hexdigest() != self.input_sha256:
            raise ValueError("Sweep raw file changed after replay selection")
        prompt, ar, mtp, ar_index, mtp_index, comparison = self._selection()
        prompt_provenance = self._prompt_provenance()
        tokenizer, tokenizer_provenance, vocabulary_provenance = self._tokenizer()
        lanes = {}
        for key, row in (("baseline", ar), ("mtp", mtp)):
            lane = deepcopy(row["measurement"]["generation_trace"])
            lane.update(display_tokens(tokenizer, lane["token_ids"]))
            _enrich_events(lane, tokenizer)
            lanes[key] = lane
        if lanes["baseline"]["eos_token_ids"] != lanes["mtp"]["eos_token_ids"]:
            raise ValueError("EOS definitions differ between saved trajectories")
        vocabulary = deepcopy(self.report.get("draft_vocabulary"))
        method_label = f"Native MTP · K={self.block_size}"
        if vocabulary:
            method_label += f" · draft shortlist {vocabulary['shortlist_size']:,}/{vocabulary['target_vocab_size']:,}"
        result = {"schema_version": 1, "prompt": prompt.get("problem") or tokenizer.decode(prompt["prompt_tokens"], skip_special_tokens=True),
                  "prompt_token_ids": list(prompt["prompt_tokens"]),
                  "max_new_tokens": len(lanes["baseline"]["token_ids"]),
                  "eos_token_ids": lanes["baseline"]["eos_token_ids"],
                  "metadata": {"dataset_index": prompt.get("index"), "prompt_index": self.prompt_index,
                      "repeat": self.repeat, "block_size": self.block_size, "max_draft_tokens_per_round": self.block_size - 1,
                      "method_label": method_label, "baseline_label": "Full-vocabulary autoregressive decode",
                      "enable_thinking": True, "sampling": "greedy", "ignore_eos": True,
                      "source": "saved mtp_sweep generation_trace; no inference, reconstructed timestamps or GIF encoding",
                      "timing_origin": "each lane's recorded backend prefill start; neighboring sequential measurements",
                      "display_policy": "offline token decoding; original events and timestamps retained",
                      "decoded_input_text": tokenizer.decode(prompt["prompt_tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False),
                      "tokenizer": tokenizer_provenance, "draft_vocabulary": vocabulary,
                      "provenance": {"raw": {"path": str(self.input), "sha256": self.input_sha256,
                                               "decompressed_json_sha256": self.document_sha256},
                                     "prompts": prompt_provenance, "draft_vocabulary": vocabulary_provenance,
                                     "pair_id": mtp["pair_id"], "baseline_run_index": ar_index, "mtp_run_index": mtp_index,
                                     "baseline_measurement_sha256": sha256(json.dumps(ar["measurement"], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                                     "mtp_measurement_sha256": sha256(json.dumps(mtp["measurement"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()}},
                  **lanes, "parity": comparison}
        if sha256(_read_bounded(self.input)).hexdigest() != self.input_sha256:
            raise ValueError("Sweep raw file changed during replay construction")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed mtp_sweep raw.json or raw.json.gz")
    parser.add_argument("--prompt-index", type=int, default=0, help="Position in protocol.prompts, not a source-dataset row ID")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--block-size", type=int, choices=range(2, 6), default=3)
    parser.add_argument("--output", type=Path, required=True, help="Replay trace.json; no GIF is encoded")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output == args.input.expanduser().resolve():
        parser.error("Replay output must not overwrite the raw sweep source")
    builder = SweepReplayBuilder(args.input, args.prompt_index, args.repeat, args.block_size)
    trace = builder.build()
    write_json(output, trace)
    print(f"Replay trace: {output}", flush=True)
    print(trace["metadata"]["method_label"], flush=True)
    print("Exact neighboring AR/MTP pair; original timestamps preserved; no inference or GIF encoding.", flush=True)
    return 0
