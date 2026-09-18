"""Measurement-Gated Provenance Attenuation: published models and comparison maps.

Fit coordinates and measurement gates on FIT data, then call an estimator's
``fit`` and source-blind ``transform``. CORAL/FEATMAP explicitly require source
identity at application. Selection and independent evaluation live outside
the estimators, preserving the paper's separation of data roles.
"""
from .baselines import CORAL, FEATMAP, Identity, LEACE
from .closed_form import ClosedFormMGPA
from .core import Endpoint, EndpointUnavailableError, permission_basis, relative_movement
from .igbp import IGBP
from .iterative import IterativeMGPA
from .serialization import load_model, save_model
from .tokens import TokenOffsetAdapter

__all__ = ["ClosedFormMGPA", "IterativeMGPA", "Identity", "LEACE", "CORAL",
           "FEATMAP", "IGBP", "TokenOffsetAdapter", "Endpoint", "EndpointUnavailableError",
           "permission_basis", "relative_movement", "save_model", "load_model"]
