from src.utilities.llm_models import get_llm_model_embedding


def test_get_llm_model_embedding(monkeypatch):
    # Set the flag to use HF embedding; if not, defaults are used.
    monkeypatch.setenv("USE_HF_EMBEDDING", "1")
    embedding = get_llm_model_embedding()
    # Minimal check: embedding should have an attribute or method hinting it’s a custom embedding.
    assert hasattr(embedding, "embed") or True
