"""LLM utilities for evaluation."""
from typing import TYPE_CHECKING

from ..config import EvaluationConfig

if TYPE_CHECKING:
    from ...src.utilities.llm_models import get_llm_model_chat
    from ...src.rag_pipeline.rag_system import RAGSystem


def get_llm_client(temperature: float = 0.1, max_tokens: int = 1000):
    """Get LLM client for evaluation."""
    # Import here to avoid circular imports
    from ...src.utilities.llm_models import get_llm_model_chat
    return get_llm_model_chat(temperature=temperature, max_tokens=max_tokens)


def get_rag_system():
    """Get RAG system for predictions."""
    # Import here to avoid circular imports
    from ...src.rag_pipeline.rag_system import RAGSystem
    return RAGSystem()