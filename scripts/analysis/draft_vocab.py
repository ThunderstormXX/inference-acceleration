"""Build a deterministic MTP draft vocabulary from calibration outputs only."""
from __future__ import annotations
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != DIRECTORY]
sys.path.insert(0, str(ROOT / "src"))

from inference_lab.optimizations.draft_vocab import build_draft_vocabulary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "docs/results/ngram-feasibility-inputs-2026-09-22.json.gz")
    parser.add_argument("--model", type=Path, default=ROOT / "models/qwen3.5-9b-mlx-4bit")
    parser.add_argument("--prefix-tokens", type=int, default=0,
                        help="Union a deterministic vocabulary ID prefix; a development-tuned capacity variant")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = args.bundle.read_bytes()
    bundle = json.loads(gzip.decompress(payload) if args.bundle.suffix == ".gz" else payload)
    tokenizer_bytes = (args.model / "tokenizer.json").read_bytes()
    model_bytes = (args.model / "config.json").read_bytes()
    manifest_hash = sha256((args.model / "download-manifest.json").read_bytes()).hexdigest()
    config = build_draft_vocabulary(bundle, json.loads(tokenizer_bytes), json.loads(model_bytes), manifest_hash,
                                    prefix_tokens=args.prefix_tokens)
    config.update(bundle_sha256=sha256(payload).hexdigest(), tokenizer_sha256=sha256(tokenizer_bytes).hexdigest(),
                  model_config_sha256=sha256(model_bytes).hexdigest(),
                  builder_sha256=sha256((ROOT / "src/inference_lab/optimizations/draft_vocab.py").read_bytes()).hexdigest())
    suffix = (f"{args.prefix_tokens // 1024}k" if args.prefix_tokens % 1024 == 0 else str(args.prefix_tokens))
    filename = f"mtp-draft-vocab-{suffix}.json" if args.prefix_tokens else "mtp-draft-vocab.json"
    output = args.output or ROOT / "configs/optimizations" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"Draft vocabulary: {config['shortlist_size']}/{config['target_vocab_size']} IDs; calibration rows100..109; prefix={args.prefix_tokens}.")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
