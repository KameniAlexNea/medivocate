from types import SimpleNamespace

# ...existing imports...
from langchain_core.documents import Document

from src.vector_store.bivector_store import VectorStoreManager


# Dummy classes to simulate external dependencies
class DummyChroma:
    def __init__(self, documents=None, **kwargs):
        self.documents = documents or []
        self.collection_name = kwargs.get("collection_name")
        self.persist_directory = kwargs.get("persist_directory")

    def add_documents(self, docs):
        self.documents.extend(docs)

    def get(self, include):
        return {
            "documents": ["dummy_doc"],
            "ids": ["1"],
            "metadatas": [{"meta": "dummy"}],
        }

    def as_retriever(self, search_kwargs):
        return SimpleNamespace(search_kwargs=search_kwargs)


class DummyBM25:
    def __init__(self, documents, tokenizer=None):
        self.documents = documents
        self.tokenizer = tokenizer


class DummyMultiQueryRetriever:
    def __init__(self, retriever, llm, include_original, prompt):
        self.retriever = retriever
        self.llm = llm
        self.include_original = include_original
        self.prompt = prompt


# Dummy DocumentLoader to simulate load_documents
class DummyDocumentLoader:
    def __init__(self, doc_dir):
        self.doc_dir = doc_dir

    def load_documents(self):
        return [Document(page_content="Test content", id="dummy", metadata={})]


# Dummy LLM for create_retriever test
class DummyLLM:
    pass


# Test: Initialize with documents
def test_initialize_with_documents(monkeypatch):
    dummy_docs = [Document(page_content="Content", id="1", metadata={})]

    monkeypatch.setattr(
        "medivocate.src.vector_store.bivector_store.Chroma.from_documents",
        lambda **kwargs: DummyChroma(documents=kwargs.get("documents")),
    )
    monkeypatch.setattr(
        "medivocate.src.vector_store.bivector_store.BM25Retriever.from_documents",
        lambda docs, tokenizer=None: DummyBM25(docs, tokenizer),
    )

    manager = VectorStoreManager(persist_directory="/tmp")
    manager._batch_process_documents(dummy_docs)

    assert manager.vs_initialized is True
    assert isinstance(manager.vector_stores["chroma"], DummyChroma)
    assert isinstance(manager.vector_stores["bm25"], DummyBM25)


# Test: Initialize without documents (loading mode)
def test_initialize_without_documents(monkeypatch):
    # Patch Chroma constructor and get method
    monkeypatch.setattr(
        "medivocate.src.vector_store.bivector_store.Chroma",
        lambda **kwargs: DummyChroma(**kwargs),
    )
    # Patch BM25Retriever.from_documents
    monkeypatch.setattr(
        "medivocate.src.vector_store.bivector_store.BM25Retriever.from_documents",
        lambda docs, tokenizer=None: DummyBM25(docs, tokenizer),
    )

    manager = VectorStoreManager(persist_directory="/tmp")
    manager.initialize_vector_store()  # no documents provided

    # After loading, bm25 should be set from dummy docs returned by get()
    assert manager.vs_initialized is True
    assert isinstance(manager.vector_stores["bm25"], DummyBM25)


# Test: Create retriever
def test_create_retriever(monkeypatch):
    dummy_bm25 = DummyBM25([])
    dummy_chroma = DummyChroma()
    manager = VectorStoreManager(persist_directory="/tmp")
    manager.vector_stores["bm25"] = dummy_bm25
    manager.vector_stores["chroma"] = dummy_chroma

    # Patch MultiQueryRetriever.from_llm to return a dummy retriever
    monkeypatch.setattr(
        "medivocate.src.vector_store.bivector_store.MultiQueryRetriever.from_llm",
        lambda retriever, llm, include_original, prompt: DummyMultiQueryRetriever(
            retriever, llm, include_original, prompt
        ),
    )

    retriever = manager.create_retriever(DummyLLM(), n_documents=5)

    assert isinstance(retriever, DummyMultiQueryRetriever)
    # Ensure BM25 retriever's k attribute is set correctly
    assert getattr(dummy_bm25, "k", None) == 5


# Test: Load and process documents using the dummy loader
def test_load_and_process_documents(monkeypatch):
    monkeypatch.setattr(
        "medivocate.src.vector_store.bivector_store.DocumentLoader",
        lambda doc_dir: DummyDocumentLoader(doc_dir),
    )
    manager = VectorStoreManager(persist_directory="/tmp")
    docs = manager.load_and_process_documents("dummy_dir")

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0].page_content == "Test content"
