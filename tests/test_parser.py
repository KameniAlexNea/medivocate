from src.utilities.parser import parse_json

# ...existing imports if any...


def test_parse_json_empty():
    result = parse_json("")
    assert result is None


def test_parse_json_malformed():
    input_str = 'text before ```json\n{"a": 1}\n``` and extra ```json\n{"b": 2}\n```'
    result = parse_json(input_str)
    assert result == {"a": 1}


def test_parse_json_direct():
    input_str = '{"key": "value"}'
    result = parse_json(input_str)
    assert result == {"key": "value"}


def test_parse_json_with_backticks():
    input_str = 'Some text\n```json\n{"num": 123}\n```'
    result = parse_json(input_str)
    assert result == {"num": 123}


def test_parse_json_invalid():
    input_str = "Not a JSON string"
    result = parse_json(input_str)
    assert result is None
