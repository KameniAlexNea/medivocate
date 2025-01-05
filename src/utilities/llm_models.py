from configparser import ConfigParser
from enum import Enum
from typing import Union
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings


llm_config = ConfigParser()

llm_config.read(Path(__file__).parent / "llm_config.ini")

ollama_config = llm_config["OLLAMA"]
groq_config = llm_config["GROQ"]


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
            model=ollama_config["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            # other params...
            base_url=ollama_config["host"],
            client_kwargs={
                "headers": {
                    "Authorization": "Bearer " + (ollama_config.get("token", ""))
                }
            },
        )
    return model_type.value(
        model=groq_config["model"],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_llm_model_embedding():
    return OllamaEmbeddings(
        model=ollama_config["embedding_model"],
        base_url=ollama_config["host"],
        client_kwargs={
            "headers": {"Authorization": "Bearer " + (ollama_config.get("token", ""))}
        },
    )
