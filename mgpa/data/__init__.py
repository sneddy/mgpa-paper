"""Participant-separated EEG data preparation; no adapter or reader fitting."""
from .prepared import ROLES, load_prepared, prepare_context, sha256, write_prepared

__all__ = ["ROLES", "load_prepared", "prepare_context", "sha256", "write_prepared"]
