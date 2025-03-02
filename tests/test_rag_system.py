from types import SimpleNamespace

import pytest

from src.rag_pipeline.rag_system import RAGSystem


# Dummy chain to simulate streaming of answers
class DummyChain:
    def stream(self, inp):
        for i in range(3):
            yield {"answer": f"Answer part {i}"}


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
