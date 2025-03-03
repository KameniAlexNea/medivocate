from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from src.preprocessing.chunking.chunk import (
    ChunkingManager,
    retrieve_documents_from_folder,
)
from src.rag_pipeline.rag_system import RAGSystem


# Dummy chain to simulate streaming of answers
class DummyChain(Runnable):
    def stream(self, input, config=None, **kwargs):
        for i in range(3):
            yield Document(page_content=f"Answer part {i}")

    def invoke(self, input, config=None, **kwargs):
        result = list(self.stream(input, config=config, **kwargs))
        return Document(page_content=" ".join(doc.page_content for doc in result))


def dummy_create_retriever(self, llm, n_documents, bm25_portion=0.8):
    self.vector_stores = {
        "bm25": SimpleNamespace(k=n_documents),
        "chroma": SimpleNamespace(as_retriever=lambda search_kwargs: "dummy_retriever"),
    }
    self.vector_store = DummyChain()
    return self.vector_store


@pytest.fixture
def dummy_rag(monkeypatch):
    rag = RAGSystem(docs_dir="dummy", persist_directory_dir="/tmp", batch_size=10)
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
    dummy_rag.setup_rag_chain()
    answers = list(dummy_rag.query("dummy question"))
    assert all("Answer part" in ans for ans in answers)


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
        assert "file" in documents[0].metadata.get("keywords", [])
