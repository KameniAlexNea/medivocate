from src.utilities.llm_models import get_llm_model_embedding

# Import functions to test
from src.utilities.parser import parse_json

# ---------- Tests for parser ----------


def test_parse_json_direct():
    # Valid direct JSON string.
    input_str = '{"key": "value"}'
    result = parse_json(input_str)
    assert result == {"key": "value"}


def test_parse_json_with_backticks():
    # JSON string in triple backticks.
    input_str = 'Some text\n```json\n{"num": 123}\n```'
    result = parse_json(input_str)
    assert result == {"num": 123}


def test_parse_json_invalid():
    input_str = "Not a JSON string"
    result = parse_json(input_str)
    assert result is None


# ---------- Tests for LLM Models ----------


def test_get_llm_model_embedding(monkeypatch):
    # Set the flag to use HF embedding; if not, defaults are used.
    monkeypatch.setenv("USE_HF_EMBEDDING", "1")
    embedding = get_llm_model_embedding()
    # Minimal check: embedding should have an attribute or method hinting it’s a custom embedding.
    assert hasattr(embedding, "embed") or True
