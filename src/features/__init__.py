"""
src/features
============
Feature engineering and class-imbalance correction.

Public API
----------
build_features  — add time, amount, and interaction features to a DataFrame
resample        — SMOTE / ADASYN / undersample the training set
"""

from src.features.feature_engineering import build_features
from src.features.resampling import resample

__all__ = ["build_features", "resample"]
