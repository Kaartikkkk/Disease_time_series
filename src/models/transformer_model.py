"""Forwarding wrapper for deep_learning.transformer_model."""
from src.models.deep_learning.transformer_model import *
try:
    from src.models.deep_learning.transformer_model import _TimeSeriesTransformerNet
except ImportError:
    _TimeSeriesTransformerNet = None
