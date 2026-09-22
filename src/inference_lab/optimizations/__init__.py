"""Opt-in execution experiments that preserve the stored model weights."""

from .affine import ScopedAffineOptimization

__all__ = ["ScopedAffineOptimization"]
