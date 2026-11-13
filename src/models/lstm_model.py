"""Forwarding wrapper for deep_learning.lstm_model."""
from src.models.deep_learning.lstm_model import *
try:
    from src.models.deep_learning.lstm_model import _LSTMNet
except ImportError:
    _LSTMNet = None

