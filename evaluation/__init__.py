"""Medivocate Evaluation Package.

This package provides tools for evaluating RAG (Retrieval-Augmented Generation)
systems through automated Q&A pair generation, prediction running, and evaluation.
"""

from .config import EvaluationConfig
from .core.data_generator import DataGenerator
from .core.evaluator import Evaluator
from .core.metrics import EvaluationMetrics
from .core.predictor import Predictor
from .models.evaluation_data import EvaluationResult, QAPair

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "EvaluationConfig",
    # Core classes
    "DataGenerator",
    "Predictor",
    "Evaluator",
    "EvaluationMetrics",
    # Data models
    "QAPair",
    "EvaluationResult",
]
