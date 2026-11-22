"""Forwarding wrapper for deep_learning.gru_model."""
from src.models.deep_learning.gru_model import *
try:
    from src.models.deep_learning.gru_model import _GRUNet
except ImportError:
    _GRUNet = None

