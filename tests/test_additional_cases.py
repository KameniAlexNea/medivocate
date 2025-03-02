from types import SimpleNamespace

import pytest

from src.preprocessing.chunking.chunk import (
    ChunkingManager,
    retrieve_documents_from_folder,
)

# Import functions/classes to test
from src.preprocessing.processor import Processor
from src.rag_pipeline.rag_system import RAGSystem
from langchain_core.runnables import Runnable

# ---------- Tests for Processor ----------


def test_processor_merge_sentences():
    text = "This is a hyphen-\nated word. And a new sentence.\nAN UPPERCASE TITLE\ncontinuing text."
    merged = Processor.merge_sentences(text)
    # Expect hyphen removed and new lines for punctuation and title separation.
    assert "hyphenated" in merged
    assert "AN UPPERCASE TITLE" in merged


def test_processor_is_valid_file():
    valid_text = "\n".join(["This is a valid line."] * 20)
    invalid_text = "\n".join(["........"] * 10)
    assert Processor.is_valid_file(valid_text) is True
    assert Processor.is_valid_file(invalid_text) is False


def test_processor_split_text_into_large_chunks():
    sample_text = ("word " * 1000).strip()
    chunks = Processor.split_text_into_large_chunks(sample_text, target_word_count=300)
    # Expect multiple chunks returned
    assert len(chunks) > 1
    total_words = sum(len(chunk.split()) for chunk in chunks)
    assert total_words >= 1000


# ---------- Tests for RAGSystem ----------


class DummyChain:
    def stream(self, inp):
        # Yield dummy tokens
        for i in range(3):
            yield {"answer": f"Answer part {i}"}


def dummy_create_retriever(self, llm, n_documents, bm25_portion=0.8):
    self.vector_stores = {
        "bm25": SimpleNamespace(k=n_documents),
        "chroma": SimpleNamespace(as_retriever=lambda search_kwargs: "dummy_retriever"),
    }
    # Dummy chain simulating MultiQueryRetriever
    self.vector_store = DummyChain()
    return self.vector_store


@pytest.fixture
def dummy_rag(monkeypatch):
    rag = RAGSystem(docs_dir="dummy", persist_directory_dir="/tmp", batch_size=10)
    # Monkey-patch initialize_vector_store and create_retriever
    monkeypatch.setattr(
        rag.vector_store_management,
        "initialize_vector_store",
        lambda documents=None: None,
    )
    monkeypatch.setattr(
        rag.vector_store_management,
        "create_retriever",
        dummy_create_retriever.__get__(rag.vector_store_management),
    )
    return rag


def test_rag_system_query(dummy_rag):
    # Setup dummy chain that yields tokens.
    dummy_rag.setup_rag_chain()
    answers = list(dummy_rag.query("dummy question"))
    assert all("Answer part" in ans for ans in answers)


# ---------- Tests for ChunkingManager's retrieve_documents_from_folder ----------


class DummyLLM:
    def __init__(self):
        pass

    def process(self, text):
        return "cleaned " + text[:10]


@pytest.fixture
def temp_text_folder(tmp_path):
    # Create a temporary folder with text files
    folder = tmp_path / "book"
    folder.mkdir()
    file1 = folder / "page1.txt"
    file2 = folder / "page2.txt"
    file1.write_text("This is sample text for file one.")
    file2.write_text("Another sample text for file two.")
    return str(folder)


@pytest.fixture
def dummy_chunking_manager():
    llm = DummyLLM()
    manager = ChunkingManager(llm=llm, chunk_size=50, chunk_overlap=10, top_n=2)
    # Override methods to avoid complex LLM calls
    manager.llm_summary = type(
        "Dummy",
        (),
        {"batch_process": lambda self, paragraphs: ["summary" for _ in paragraphs]},
    )()
    manager.llm_keyword = type(
        "Dummy",
        (),
        {
            "batch_process": lambda self, paragraphs: [
                ["kw1", "kw2"] for _ in paragraphs
            ]
        },
    )()
    manager.llm_clean = type(
        "Dummy", (), {"process": lambda self, text: "cleaned " + text}
    )()
    manager.llm_category = type(
        "Dummy", (), {"process": lambda self, text: "contenu"}
    )()
    return manager


def test_retrieve_documents_from_folder(temp_text_folder, dummy_chunking_manager):
    documents = retrieve_documents_from_folder(
        dummy_chunking_manager,
        temp_text_folder,
        use_llm_cleaning=True,
        use_llm_for_keywords=False,
        summarize_before_chunk=False,
        check_text_validity=False,
        llm_check_text_validity=False,
        verbose=False,
        target_word_count=10,
    )
    # Expect documents list is not empty and document metadata has keywords.
    assert isinstance(documents, list)
    if documents:
        assert "kw1" in documents[0].metadata.get("keywords", [])
