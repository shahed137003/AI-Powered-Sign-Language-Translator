"""
Core package for real-time ASL recognition system.

This package exposes the main components of the system:
- ASL real-time pipeline
- Model wrapper (TCN inference)
- LLM translator
- Keypoint extractor

This allows clean imports like:
    from core import ASLPipeline
"""

from .pipeline import ASLPipeline
from .model import ModelWrapper
from .llm import LLMTranslator
from .keypoints import extract_keypoints

__all__ = [
    "ASLPipeline",
    "ModelWrapper",
    "LLMTranslator",
    "extract_keypoints",
]