"""Acquire the pinned native Qwen3.5 MTP head without duplicate target/tokenizer files."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .download import ModelDownloader, PROJECT_ROOT


MTP_ID = "mlx-community/Qwen3.5-9B-MTP-4bit"
MTP_REVISION = "222dfd2c23fc9518d7b817e4f8e0cb0571787489"
MTP_WEIGHT_BYTES = 136_884_332
MTP_WEIGHT_SHA256 = "ff2f9298bb78a0f015a9998fecda6b5d40c2995d9ecd55e65a206ee5e6d6f897"
MTP_FILES = ("config.json", "model.safetensors", "model.safetensors.index.json", "README.md", ".gitattributes")


@dataclass(frozen=True)
class MTPDownloadConfig:
    output_dir: Path = PROJECT_ROOT / "models/qwen3.5-9b-mtp-4bit"
    max_workers: int = 2
    xet_download_concurrency: int = 16


class MTPDownloader:
    """Download a small, standalone MTP head and verify its pinned weight digest."""

    def __init__(self, config: MTPDownloadConfig):
        self.config = config

    def run(self) -> dict:
        if self.config.max_workers < 1 or self.config.xet_download_concurrency < 1:
            raise ValueError("Download worker counts must be positive")
        output = self.config.output_dir.resolve()
        protected = {PROJECT_ROOT / "models/qwen3.5-9b-mlx-4bit", PROJECT_ROOT / "models/qwen3.5-9b-dflash"}
        if output in {path.resolve() for path in protected}:
            raise ValueError("The MTP head must not overwrite an existing target or DFlash directory")
        manifest_path = output / "download-manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing.get("repo_id") != MTP_ID or existing.get("resolved_revision") != MTP_REVISION:
                raise ValueError("The output directory contains a different checkpoint; choose an empty MTP directory")
        os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
        os.environ.setdefault("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", str(self.config.xet_download_concurrency))
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(MTP_ID, revision=MTP_REVISION, files_metadata=True)
        if info.sha != MTP_REVISION:
            raise RuntimeError("Hugging Face returned an unexpected MTP revision")
        siblings = {item.rfilename: item for item in info.siblings or []}
        if set(MTP_FILES) - siblings.keys():
            raise RuntimeError("Pinned MTP snapshot is missing a required file")
        weight = siblings["model.safetensors"]
        if weight.size != MTP_WEIGHT_BYTES or getattr(weight.lfs, "sha256", None) != MTP_WEIGHT_SHA256:
            raise RuntimeError("MTP weight source metadata differs from the known pinned checkpoint")
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.unlink(missing_ok=True)
        print(f"Downloading {MTP_ID}@{MTP_REVISION} -> {output}", flush=True)
        snapshot_download(MTP_ID, revision=MTP_REVISION, local_dir=output,
                          allow_patterns=list(MTP_FILES), max_workers=self.config.max_workers)
        files = []
        for name in MTP_FILES:
            item, path = siblings[name], output / name
            if not path.is_file() or path.stat().st_size != item.size:
                raise RuntimeError(f"Missing or incomplete MTP file: {name}")
            checksum = ModelDownloader._sha256(path)
            expected = getattr(item.lfs, "sha256", None) if item.lfs else None
            if expected and checksum != expected:
                raise RuntimeError(f"Source SHA256 mismatch for {name}")
            if name == "model.safetensors" and checksum != MTP_WEIGHT_SHA256:
                raise RuntimeError("MTP weight SHA256 differs from the known pinned checkpoint")
            files.append({"path": name, "bytes": path.stat().st_size, "sha256": checksum,
                          "source_lfs_sha256": expected})
        with (output / "model.safetensors").open("rb") as stream:
            header_length = int.from_bytes(stream.read(8), "little")
            if not 0 < header_length <= 16 * 1024 * 1024:
                raise RuntimeError("Invalid safetensors header size")
            header = json.loads(stream.read(header_length))
        tensor_names = set(header) - {"__metadata__"}
        dtypes = sorted({header[name]["dtype"] for name in tensor_names})
        index = json.loads((output / "model.safetensors.index.json").read_text())
        if set(index["weight_map"]) != tensor_names or set(index["weight_map"].values()) != {"model.safetensors"}:
            raise RuntimeError("MTP safetensors index does not match the downloaded tensors")
        model_config = json.loads((output / "config.json").read_text())
        if model_config.get("model_type") != "qwen3_5_mtp":
            raise RuntimeError("Downloaded configuration is not the expected native Qwen3.5 MTP head")
        manifest = {
            "schema_version": 1, "repo_id": MTP_ID,
            "requested_revision": MTP_REVISION, "resolved_revision": info.sha,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "format": "Native Qwen3.5 MLX MTP head; tensor dtypes: " + ", ".join(dtypes),
            "model_type": model_config["model_type"],
            "quantization": model_config.get("quantization", model_config.get("quantization_config")),
            "tensor_dtypes": dtypes, "tensor_count": len(tensor_names), "scope": "selected_mtp_head_files",
            "allowed_files": list(MTP_FILES), "local_dir": str(output),
            "tokenizer": "Reuses the target model tokenizer; mlx_vlm load_drafter loads only the MTP head",
            "total_bytes": sum(item["bytes"] for item in files), "files": files,
            "transfer": {"xet_fixed_download_concurrency": os.environ["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"],
                         "xet_chunk_cache_size_bytes": os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"]},
        }
        ModelDownloader._write_manifest(manifest_path, manifest)
        print(f"MTP head ready: {len(files)} verified files, {manifest['total_bytes']:,} bytes; dtypes={dtypes}", flush=True)
        print(f"Manifest: {manifest_path}", flush=True)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MTPDownloadConfig.output_dir)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--xet-download-concurrency", type=int, default=16)
    MTPDownloader(MTPDownloadConfig(**vars(parser.parse_args()))).run()


if __name__ == "__main__":
    main()
