"""
src/data
========
Data loading, cleaning, scaling, and stratified splitting.

Public API
----------
load_data           — load creditcard.csv or generate synthetic fallback
preprocess_pipeline — full offline preprocessing returning train/val/test splits
"""

from src.data.loader import load_data
from src.data.preprocessing import preprocess_pipeline

__all__ = ["load_data", "preprocess_pipeline"]
