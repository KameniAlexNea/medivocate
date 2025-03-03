import pytest

from src.preprocessing.chunking.chunk import (
    ChunkingManager,
    retrieve_documents_from_folder,
)


# Dummy LLM and related patch for chunking tests
class DummyLLM:
    def process(self, text):
        return "cleaned " + text[:10]


@pytest.fixture
def dummy_chunking_manager():
    llm = DummyLLM()
    manager = ChunkingManager(llm=llm, chunk_size=50, chunk_overlap=10, top_n=2)
    manager.llm_keyword = type(
        "Dummy",
        (),
        {"batch_process": lambda self, paragraphs: [["dummy_kw"] for _ in paragraphs]},
    )()
    manager.llm_clean = type(
        "Dummy", (), {"process": lambda self, text: "cleaned " + text}
    )()
    manager.llm_summary = type(
        "Dummy",
        (),
        {"batch_process": lambda self, paragraphs: [para[:20] for para in paragraphs]},
    )()
    manager.llm_category = type(
        "Dummy", (), {"process": lambda self, text: "contenu"}
    )()
    return manager


def test_retrieve_documents_empty_folder(tmp_path, dummy_chunking_manager):
    empty_folder = tmp_path / "empty_folder"
    empty_folder.mkdir()
    docs = retrieve_documents_from_folder(
        dummy_chunking_manager, str(empty_folder), verbose=False, target_word_count=100
    )
    assert docs == []


def test_retrieve_documents_no_valid_text(
    tmp_path, dummy_chunking_manager, monkeypatch
):
    folder = tmp_path / "invalid"
    folder.mkdir()
    file_path = folder / "page1.txt"
    file_path.write_text("........\n" * 5)
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
    assert docs == []


def test_retrieve_documents_with_llm_cleaning(tmp_path, dummy_chunking_manager):
    folder = tmp_path / "cleaning"
    folder.mkdir()
    file_path = folder / "doc.txt"
    original_text = "This text requires cleaning and further processing."
    file_path.write_text(original_text)
    docs = retrieve_documents_from_folder(
        dummy_chunking_manager,
        str(folder),
        use_llm_cleaning=True,
        use_llm_for_keywords=False,
        summarize_before_chunk=False,
        check_text_validity=False,
        llm_check_text_validity=False,
        verbose=False,
        target_word_count=3,
    )
    assert isinstance(docs, list)
    if docs:
        # llm_clean is set up to prepend "cleaned " to the input text
        assert docs[0].page_content.startswith("cleaned ")


def test_retrieve_documents_with_keywords_enabled(tmp_path, dummy_chunking_manager):
    folder = tmp_path / "keywords"
    folder.mkdir()
    file_path = folder / "doc.txt"
    file_path.write_text("Some sample content for keyword extraction.")
    docs = retrieve_documents_from_folder(
        dummy_chunking_manager,
        str(folder),
        use_llm_cleaning=False,
        use_llm_for_keywords=True,
        summarize_before_chunk=False,
        check_text_validity=False,
        llm_check_text_validity=False,
        verbose=False,
        target_word_count=3,
    )
    assert isinstance(docs, list)
    if docs:
        # When llm_for_keywords is True, the dummy llm_keyword should add "dummy_kw"
        assert "dummy_kw" in docs[0].metadata.get("keywords", [])
