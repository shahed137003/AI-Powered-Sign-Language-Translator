"""Preprocessing package (v3 global-mean root+scale + safehands)."""

from .pipeline_v3_with_clip_logging import preprocess_sequence_global

__all__ = ["preprocess_sequence_global"]
