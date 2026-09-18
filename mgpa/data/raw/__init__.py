"""Optional raw EEG acquisition and frozen feature extraction.

Nothing is downloaded or encoded on import. Call acquire with explicit local
raw-data and checkpoint paths, then prepare the resulting pinned receipt.
"""
from .pipeline import acquire

__all__ = ["acquire"]
