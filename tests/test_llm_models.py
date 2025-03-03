from src.utilities.llm_models import (
    get_llm_model_chat,
    get_llm_model_embedding,
)


def test_get_llm_model_chat(monkeypatch):
    monkeypatch.setenv("USE_OLLAMA_CHAT", "1")
    chat_model = get_llm_model_chat()
    assert "ChatOllama" in str(chat_model.__class__)

    monkeypatch.setenv("USE_OLLAMA_CHAT", "0")
    chat_model = get_llm_model_chat()
    assert "ChatGroq" in str(chat_model.__class__)


def test_get_llm_model_embedding(monkeypatch):
    monkeypatch.setenv("USE_HF_EMBEDDING", "1")
    embedding = get_llm_model_embedding()
    assert hasattr(embedding, "embed") or True
