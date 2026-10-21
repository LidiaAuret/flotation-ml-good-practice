"""Small helpers for honest evaluation of process ML models.

Kept deliberately thin: the point of the examples is the evaluation design,
not the library.
"""
from .splits import random_split, blocked_split, lag1_autocorr, n_eff_ratio
from .models import make_model, n_parameters
from .metrics import scores

__all__ = ["random_split", "blocked_split", "lag1_autocorr", "n_eff_ratio",
           "make_model", "n_parameters", "scores"]
