import os
import shutil

import pytest

from src.preprocessing.chunking.chunk import (
    ChunkingManager,
    retrieve_documents_from_folder,
)
from src.preprocessing.processor import Processor
from src.utilities.load_data import download_and_prepare_data

# ---------- Additional Tests for load_data ----------


def test_download_and_prepare_data_zip_error(monkeypatch, tmp_path):
    # Simulate zip extraction error by creating an empty zip file.
    zip_path = tmp_path / "empty.zip"
    zip_path.write_bytes(b"")
    extract_to = tmp_path / "extract"
    target_folder = tmp_path / "target"

    # Override subprocess.run to do nothing
    monkeypatch.setattr("subprocess.run", lambda args, check: None)
    # Override os.remove and shutil.move to bypass actual file system operations
    monkeypatch.setattr(os, "remove", lambda x: None)
    monkeypatch.setattr(shutil, "move", lambda src, dst: None)

    # download_and_prepare_data should handle zip error and log exception.
    # Here, we catch exception by not raising so function completes.
    try:
        download_and_prepare_data(
            "fake_url", str(zip_path), str(extract_to), str(target_folder)
        )
    except Exception:
        pytest.fail("download_and_prepare_data raised Exception unexpectedly!")


# ---------- Additional Tests for ChunkingManager and retrieval ----------


class DummyLLM:
    def process(self, text):
        return "cleaned " + text[:10]


@pytest.fixture
def dummy_chunking_manager():
    llm = DummyLLM()
    manager = ChunkingManager(llm=llm, chunk_size=50, chunk_overlap=10, top_n=2)
    # Override keyword extraction to return fixed keywords.
    manager.llm_keyword = type(
        "Dummy",
        (),
        {"batch_process": lambda self, paragraphs: [["dummy_kw"] for _ in paragraphs]},
    )()
    # Override cleaning to return modified text.
    manager.llm_clean = type(
        "Dummy", (), {"process": lambda self, text: "cleaned " + text}
    )()
    # Override summary to simply return the first 20 characters.
    manager.llm_summary = type(
        "Dummy",
        (),
        {"batch_process": lambda self, paragraphs: [para[:20] for para in paragraphs]},
    )()
    # Override category to always return valid indicator.
    manager.llm_category = type(
        "Dummy", (), {"process": lambda self, text: "contenu"}
    )()
    return manager


def test_retrieve_documents_empty_folder(tmp_path, dummy_chunking_manager):
    # Create an empty folder.
    empty_folder = tmp_path / "empty_folder"
    empty_folder.mkdir()
    docs = retrieve_documents_from_folder(
        dummy_chunking_manager, str(empty_folder), verbose=False, target_word_count=100
    )
    # Expect no documents if folder has no .txt files.
    assert docs == []


def test_retrieve_documents_no_valid_text(
    tmp_path, dummy_chunking_manager, monkeypatch
):
    # Create a folder with a .txt file containing invalid text.
    folder = tmp_path / "invalid"
    folder.mkdir()
    file_path = folder / "page1.txt"
    file_path.write_text("........\n" * 5)

    # Force check_text_validity to be True and llm_check_text_validity to return non-valid.
    monkeypatch.setattr(
        dummy_chunking_manager.llm_category, "process", lambda text: "invalid"
    )

    docs = retrieve_documents_from_folder(
        dummy_chunking_manager,
        str(folder),
        check_text_validity=True,
        llm_check_text_validity=True,
        verbose=False,
        target_word_count=50,
    )
    # Expect empty list when text is deemed invalid.
    assert docs == []
