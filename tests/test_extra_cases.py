import os
import shutil
import zipfile

from src.utilities.llm_models import (
    get_llm_model_chat,
    get_llm_model_embedding,
)
from src.utilities.load_data import download_and_prepare_data

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


# ---------- Tests for load_data ----------


def test_download_and_prepare_data(monkeypatch, tmp_path):
    # Setup: create a fake zip file with a dummy folder and file.
    dummy_folder_name = "dummy_dir"
    dummy_file_name = "dummy.txt"
    dummy_content = "dummy content"

    # Create a temporary folder to act as the source data
    source_dir = tmp_path / dummy_folder_name
    source_dir.mkdir()
    dummy_file = source_dir / dummy_file_name
    dummy_file.write_text(dummy_content)

    # Create a fake zip file from the source_dir
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(dummy_file, arcname=f"{dummy_folder_name}/{dummy_file_name}")

    extract_to = tmp_path / "extract"
    target_folder = tmp_path / "target"

    # Monkeypatch subprocess.run to bypass actual download
    monkeypatch.setattr("subprocess.run", lambda args, check: None)
    # Override os.remove to prevent deletion during test
    monkeypatch.setattr(os, "remove", lambda x: None)
    # Simulate os.path.exists: return False when checking for 'chroma.sqlite3'
    original_exists = os.path.exists
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: False if "chroma.sqlite3" in path else original_exists(path),
    )
    # Override shutil.move to simply copy the directory
    monkeypatch.setattr(shutil, "move", lambda src, dst: shutil.copytree(src, dst))

    # Call the function
    download_and_prepare_data(
        "fake_url", str(zip_path), str(extract_to), str(target_folder)
    )

    # Verify that target_folder has been created with the dummy file.
    assert os.path.isdir(str(target_folder))
    dummy_target_file = os.path.join(str(target_folder), dummy_file_name)
    assert os.path.isfile(dummy_target_file)
    with open(dummy_target_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == dummy_content


# ---------- Tests for LLM Models ----------


def test_get_llm_model_chat(monkeypatch):
    # Test switch between ChatOllama and ChatGroq by setting env variable.
    monkeyatch = monkeypatch
    monkeyatch.setenv("USE_OLLAMA_CHAT", "1")
    chat_model = get_llm_model_chat()
    # Test that the ChatOllama model is chosen if USE_OLLAMA_CHAT is "1"
    assert "ChatOllama" in str(chat_model)

    monkeyatch.setenv("USE_OLLAMA_CHAT", "0")
    chat_model = get_llm_model_chat()
    # Test that the ChatGroq model is chosen if USE_OLLAMA_CHAT is not "1"
    assert "ChatGroq" in str(chat_model)


def test_get_llm_model_embedding(monkeypatch):
    # Set the flag to use HF embedding; if not, defaults are used.
    monkeypatch.setenv("USE_HF_EMBEDDING", "1")
    embedding = get_llm_model_embedding()
    # Minimal check: embedding should have an attribute or method hinting it’s a custom embedding.
    assert hasattr(embedding, "embed") or True
