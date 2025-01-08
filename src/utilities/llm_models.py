import os
from enum import Enum

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings


class LLMModel(Enum):
    OLLAMA = ChatOllama
    GROQ = ChatGroq


def get_llm_model_chat(temperature=0.01, max_tokens=None):
    if str(os.getenv("USE_OLLAMA_CHAT")) == "1" and "localhost" not in str(
        os.getenv("OLLAMA_HOST")
    ):
        return ChatOllama(
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
    return ChatGroq(
        model=os.getenv("GROQ_MODEL_NAME"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_llm_model_embedding():
    if str(os.getenv("USE_HF_EMBEDDING")) == "1":
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
