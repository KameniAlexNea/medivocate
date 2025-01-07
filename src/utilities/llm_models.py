import os
from enum import Enum
from typing import Union

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class LLMModel(Enum):
    OLLAMA = ChatOllama
    GROQ = ChatGroq


def get_llm_model_chat(
    model_type: Union[str, LLMModel], temperature=0, max_tokens=None
):
    if isinstance(model_type, str):
        model_type = LLMModel[model_type.upper()]
    if model_type == LLMModel.OLLAMA:
        return model_type.value(
            model=os.getenv("OLLAMA_MODEL"),
            temperature=temperature,
            max_tokens=max_tokens,
            # other params...
            base_url=os.getenv("OLLAMA_HOST"),
            client_kwargs={
                "headers": {
                    "Authorization": "Bearer " + (os.getenv("OLLAMA_TOKEN") or "")
                }
            },
        )
    return model_type.value(
        model=os.getenv("GROQ_MODEL_NAME"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_llm_model_embedding():
    if os.getenv("USE_HF"):
        return HuggingFaceEmbeddings(
            model_name=os.getenv("HF_MODEL"),  # You can replace with any HF model
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return OllamaEmbeddings(
        model=os.getenv("OLLAM_EMB"),
        base_url=os.getenv("OLLAMA_HOST"),
        client_kwargs={
            "headers": {"Authorization": "Bearer " + (os.getenv("OLLAMA_TOKEN") or "")}
        },
    )
