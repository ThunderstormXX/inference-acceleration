"""Download only the pinned original-precision DFlash draft checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .download import ModelDownloader, PROJECT_ROOT


DRAFT_ID = "z-lab/Qwen3.5-9B-DFlash"
DRAFT_REVISION = "5fc3b3d474760f18c516db87d84c37edbfd3ede6"
DRAFT_WEIGHT_BYTES = 2_583_816_465
DRAFT_WEIGHT_SHA256 = "0a42274b32554f48de1faa0d42824e9c2ceda649c30ae0a731cddf410dd698c7"
DRAFT_FILES = ("config.json", "model.safetensors", "README.md", ".gitattributes")


@dataclass(frozen=True)
class DraftDownloadConfig:
    output_dir: Path = PROJECT_ROOT / "models/qwen3.5-9b-dflash"
    max_workers: int = 2
    xet_download_concurrency: int = 16


class DraftDownloader:
    """Download four explicit files and verify source/LFS and known weight hashes."""

    def __init__(self, config: DraftDownloadConfig):
        self.config = config

    def run(self) -> dict:
        if self.config.max_workers < 1 or self.config.xet_download_concurrency < 1:
            raise ValueError("Download worker counts must be positive")
        output = self.config.output_dir.resolve()
        if output == (PROJECT_ROOT / "models/qwen3.5-9b-mlx-4bit").resolve():
            raise ValueError("The draft must not overwrite the target model directory")
        manifest_path = output / "download-manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing.get("repo_id") != DRAFT_ID or existing.get("resolved_revision") != DRAFT_REVISION:
                raise ValueError("The output directory contains a different checkpoint; choose an empty draft directory")
        os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
        os.environ.setdefault("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", str(self.config.xet_download_concurrency))
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(DRAFT_ID, revision=DRAFT_REVISION, files_metadata=True)
        if info.sha != DRAFT_REVISION:
            raise RuntimeError("Hugging Face returned an unexpected draft revision")
        siblings = {item.rfilename: item for item in info.siblings or []}
        if set(DRAFT_FILES) - siblings.keys():
            raise RuntimeError("Pinned draft snapshot is missing a required file")
        if siblings["model.safetensors"].size != DRAFT_WEIGHT_BYTES:
            raise RuntimeError("Pinned draft weight size differs from the expected checkpoint")
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.unlink(missing_ok=True)
        print(f"Downloading {DRAFT_ID}@{DRAFT_REVISION} -> {output}", flush=True)
        snapshot_download(DRAFT_ID, revision=DRAFT_REVISION, local_dir=output,
                          allow_patterns=list(DRAFT_FILES), max_workers=self.config.max_workers)
        files = []
        for name in DRAFT_FILES:
            item, path = siblings[name], output / name
            if not path.is_file() or path.stat().st_size != item.size:
                raise RuntimeError(f"Missing or incomplete draft file: {name}")
            checksum = ModelDownloader._sha256(path)
            expected = getattr(item.lfs, "sha256", None) if item.lfs else None
            if expected and checksum != expected:
                raise RuntimeError(f"Source SHA256 mismatch for {name}")
            if name == "model.safetensors" and checksum != DRAFT_WEIGHT_SHA256:
                raise RuntimeError("Draft weight SHA256 differs from the known pinned checkpoint")
            files.append({"path": name, "bytes": path.stat().st_size, "sha256": checksum})
        with (output / "model.safetensors").open("rb") as stream:
            header_length = int.from_bytes(stream.read(8), "little")
            if not 0 < header_length <= 16 * 1024 * 1024:
                raise RuntimeError("Invalid safetensors header size")
            header = json.loads(stream.read(header_length))
        dtypes = sorted({tensor["dtype"] for name, tensor in header.items() if name != "__metadata__"})
        manifest = {
            "schema_version": 1, "repo_id": DRAFT_ID,
            "requested_revision": DRAFT_REVISION, "resolved_revision": info.sha,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "format": "Original DFlash draft safetensors; tensor dtypes: " + ", ".join(dtypes),
            "tensor_dtypes": dtypes, "scope": "selected_draft_files",
            "allowed_files": list(DRAFT_FILES), "local_dir": str(output),
            "total_bytes": sum(item["bytes"] for item in files), "files": files,
            "transfer": {"xet_fixed_download_concurrency": os.environ["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"],
                         "xet_chunk_cache_size_bytes": os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"]},
        }
        ModelDownloader._write_manifest(manifest_path, manifest)
        print(f"Draft ready: {len(files)} verified files, {manifest['total_bytes']:,} bytes; dtypes={dtypes}", flush=True)
        print(f"Manifest: {manifest_path}", flush=True)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DraftDownloadConfig.output_dir)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--xet-download-concurrency", type=int, default=16)
    DraftDownloader(DraftDownloadConfig(**vars(parser.parse_args()))).run()


if __name__ == "__main__":
    main()
