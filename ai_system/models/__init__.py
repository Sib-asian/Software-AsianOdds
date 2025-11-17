"""
AI Models Module
================

Contiene tutti i modelli predittivi per l'Ensemble.

Modelli disponibili:
- XGBoostPredictor: Gradient boosting con feature engineering avanzato
- LSTMPredictor: Recurrent neural network per sequenze temporali
- MetaLearner: Neural network per peso dinamico modelli
- EnsembleMetaModel: Orchestratore principale che combina tutti i modelli
"""

from pathlib import Path

# Version
__version__ = "1.0.0"

# Models directory
MODELS_DIR = Path(__file__).parent

# Import models
from .xgboost_predictor import XGBoostPredictor
from .lstm_predictor import LSTMPredictor
from .meta_learner import MetaLearner
from .ensemble import EnsembleMetaModel

__all__ = [
    'MODELS_DIR',
    'XGBoostPredictor',
    'LSTMPredictor',
    'MetaLearner',
    'EnsembleMetaModel',
]
