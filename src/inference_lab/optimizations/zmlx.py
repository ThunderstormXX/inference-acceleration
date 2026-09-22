"""Pinned ZMLX activation/DeltaNet ablations with complete module restoration."""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
ZMLX_COMMIT = "d8cd0d88d4299ca6821d706f9d5b4a3520d688f5"


def load_zmlx():
    checkout = ROOT / ".cache/external/ZMLX"
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True).strip()
    if head != ZMLX_COMMIT or dirty:
        raise ValueError("Run scripts/setup/speed_candidates.sh for the pinned clean ZMLX checkout")
    source = str(checkout / "src")
    sys.path.insert(0, source)
    try:
        from zmlx.patch import patch
    finally:
        sys.path.remove(source)
    return patch


def _copy_containers(value):
    # Preserve Module/tensor identity while freezing plain mutable containers.
    if type(value) is dict:
        return {key: _copy_containers(child) for key, child in value.items()}
    if type(value) is list:
        return [_copy_containers(child) for child in value]
    if type(value) is tuple:
        return tuple(_copy_containers(child) for child in value)
    return value


@contextmanager
def scoped_zmlx(model, patterns):
    """Only these patterns mutate existing module call paths, never weights.

    Upstream unpatch does not reliably walk list-held layers. Restore the actual
    module objects, classes and shallow attribute mappings instead. No tensor is
    copied. Temporary state and dynamic classes disappear before the next AR run.
    """
    if not patterns or any(p not in ("deltanet", "swiglu_mlp") for p in patterns):
        raise ValueError("Only DeltaNet and SwiGLU are supported by this scoped experiment")
    patch = load_zmlx()
    modules = [module for _, module in model.named_modules()]
    snapshots = [(module, type(module), _copy_containers(dict(module)), _copy_containers(dict(module.__dict__))) for module in modules]
    try:
        patch(model, patterns=patterns)
        result = asdict(model._zmlx_patch_result)
        result.update({"repository": "https://github.com/Hmbown/ZMLX", "commit": ZMLX_COMMIT,
                       "patterns": patterns, "restoration": "all original module classes and shallow mappings"})
        if {id(module) for _, module in model.named_modules()} != {id(module) for module in modules}:
            raise RuntimeError("ZMLX replaced modules; this context only supports in-place patterns")
        if not result["patched_count"]:
            raise RuntimeError("No modules patched; not a valid speed experiment")
        yield result
    finally:
        for module, cls, mapping, attributes in reversed(snapshots):
            module.__class__ = cls
            dict.clear(module)
            dict.update(module, mapping)
            module.__dict__.clear()
            module.__dict__.update(attributes)
