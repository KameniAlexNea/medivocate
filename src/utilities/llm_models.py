import os
from enum import Enum

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .embedding import CustomEmbedding


class LLMModel(Enum):
    OLLAMA = "ChatOllama"
    GROQ = "ChatGroq"


def get_llm_model_chat(temperature=0.01, max_tokens: int = None):
    """Get a chat language model based on environment configuration.

    Supports both Ollama and Groq models based on USE_OLLAMA_CHAT environment variable.

    Args:
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens to generate

    Returns:
        Configured chat model instance

    Raises:
        ValueError: If required environment variables are not set
    """
    if str(os.getenv("USE_OLLAMA_CHAT")) == "1":
        model = os.getenv("OLLAMA_MODEL")
        if not model:
            raise ValueError("OLLAMA_MODEL environment variable must be set when USE_OLLAMA_CHAT=1")
        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
        )

    # Default to Groq
    model = os.getenv("GROQ_MODEL_NAME")
    if not model:
        raise ValueError("GROQ_MODEL_NAME environment variable must be set")
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_llm_model_embedding():
    """Get an embedding model based on environment configuration.

    Supports both custom HuggingFace embeddings and Ollama embeddings.

    Returns:
        Configured embedding model instance

    Raises:
        ValueError: If required environment variables are not set
    """
    if str(os.getenv("USE_HF_EMBEDDING")) == "1":
        return CustomEmbedding()

    # Default to Ollama embeddings
    model = os.getenv("OLLAM_EMB")
    if not model:
        raise ValueError("OLLAM_EMB environment variable must be set for Ollama embeddings")
    return OllamaEmbeddings(
        model=model,
        base_url=(
            os.getenv("OLLAMA_HOST") if os.getenv("OLLAMA_HOST") is not None else None
        ),
        client_kwargs=(
            {
                "headers": {
                    "Authorization": "Bearer " + (os.getenv("OLLAMA_TOKEN") or "")
                }
            }
            if os.getenv("OLLAMA_HOST") is not None
            else None
        ),
    )
