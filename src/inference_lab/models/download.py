"""Reproducible downloads of the original MLX snapshot, reusing complete files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "mlx-community/Qwen3.5-9B-4bit"
MODEL_REVISION = "8b2b98c00a6b4d291155e4890773ca8f769aee53"


@dataclass(frozen=True)
class ModelDownloadConfig:
    repo_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    output_dir: Path = PROJECT_ROOT / "models/qwen3.5-9b-mlx-4bit"
    max_workers: int = 2
    metadata_only: bool = False
    xet_download_concurrency: int = 16


class ModelDownloader:
    """Keep one local snapshot, pin its commit and verify downloaded files."""

    def __init__(self, config: ModelDownloadConfig) -> None:
        self.config = config

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def run(self) -> dict[str, Any]:
        # local_dir downloads directly into the project instead of first copying
        # the same multi-GB files through the global Hub blob cache.
        os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
        # Xet 1.6's adaptive controller starts with one stream; high round-trip
        # times can keep it there. A moderate fixed stream count avoids that
        # bottleneck without HIGH_PERFORMANCE's much larger memory buffers.
        if self.config.xet_download_concurrency < 1:
            raise ValueError("xet_download_concurrency must be positive")
        os.environ.setdefault("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", str(self.config.xet_download_concurrency))
        from huggingface_hub import HfApi, snapshot_download

        if self.config.max_workers < 1:
            raise ValueError("max_workers must be positive")
        info = HfApi().model_info(
            self.config.repo_id,
            revision=self.config.revision,
            files_metadata=True,
        )
        if not info.sha:
            raise RuntimeError("Hugging Face did not return a model commit SHA")
        output = self.config.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / ("metadata-manifest.json" if self.config.metadata_only else "download-manifest.json")
        # A manifest is a completion marker, never leave an old completion
        # marker describing a partially updated snapshot.
        manifest_path.unlink(missing_ok=True)
        print(f"Downloading {self.config.repo_id}@{info.sha} -> {output}", flush=True)
        snapshot_download(
            repo_id=self.config.repo_id,
            revision=info.sha,
            local_dir=output,
            max_workers=self.config.max_workers,
            ignore_patterns=["*.safetensors"] if self.config.metadata_only else None,
        )
        files = []
        for sibling in sorted(info.siblings or [], key=lambda item: item.rfilename):
            if self.config.metadata_only and sibling.rfilename.endswith(".safetensors"):
                continue
            path = output / sibling.rfilename
            if not path.is_file():
                raise RuntimeError(f"Snapshot is incomplete: missing {sibling.rfilename}")
            size = path.stat().st_size
            if sibling.size is not None and size != sibling.size:
                raise RuntimeError(f"Size mismatch for {sibling.rfilename}")
            checksum = self._sha256(path)
            expected = getattr(sibling.lfs, "sha256", None) if sibling.lfs else None
            if expected and expected != checksum:
                raise RuntimeError(f"SHA256 mismatch for {sibling.rfilename}")
            files.append({"path": sibling.rfilename, "bytes": size, "sha256": checksum})
        manifest = {
            "schema_version": 1,
            "repo_id": self.config.repo_id,
            "requested_revision": self.config.revision,
            "resolved_revision": info.sha,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "format": "MLX 4-bit safetensors; not a PyTorch 4-bit checkpoint",
            "scope": "metadata_and_tokenizer" if self.config.metadata_only else "complete_snapshot",
            "transfer": {
                "xet_fixed_download_concurrency": os.environ["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"],
                "xet_chunk_cache_size_bytes": os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"],
                "hub_note": "Completed files are reused. Hub 1.32 uses process-unique temporary files; interrupted partial files are not resumed across processes.",
            },
            "local_dir": str(output),
            "total_bytes": sum(entry["bytes"] for entry in files),
            "files": files,
        }
        self._write_manifest(manifest_path, manifest)
        label = "Model metadata ready" if self.config.metadata_only else "Model ready"
        print(f"{label}: {len(files)} files, {manifest['total_bytes']:,} bytes", flush=True)
        print(f"Manifest: {manifest_path}", flush=True)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output-dir", type=Path, default=ModelDownloadConfig.output_dir)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--metadata-only", action="store_true", help="Fetch configuration/tokenizer without weights; does not mark the model ready")
    parser.add_argument("--xet-download-concurrency", type=int, default=16)
    args = parser.parse_args()
    ModelDownloader(ModelDownloadConfig(**vars(args))).run()


if __name__ == "__main__":
    main()
