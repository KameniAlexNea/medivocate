from typing import List

from langchain.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..config import VectorStoreConfig
from .base_vector_store import BaseVectorStoreManager
from .document_loader import DocumentLoader
from .prompts import DEFAULT_QUERY_PROMPT


class VectorStoreManager(BaseVectorStoreManager):
    """
    Manages vector store initialization, updates, and retrieval.
    """

    def __init__(self, persist_directory: str, batch_size: int = 64):
        """
        Initializes the VectorStoreManager with the given parameters.

        Args:
            persist_directory (str): Directory to persist the vector store.
            batch_size (int): Number of documents to process in each batch.
        """
        super().__init__(persist_directory, batch_size)
        self.vector_stores: dict[str, Chroma] = {"chroma": None}

    def _initialize_chroma_store(self, documents: List[Document]):
        """
        Initialize Chroma vector store with first batch.

        Args:
            documents (List[Document]): First batch of documents.
        """
        self.vector_stores["chroma"] = Chroma.from_documents(
            collection_name=self.collection_name,
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def _add_chroma_documents(self, documents: List[Document]):
        """
        Add documents to existing Chroma vector store.

        Args:
            documents (List[Document]): Documents to add.
        """
        self.vector_stores["chroma"].add_documents(documents)

    def initialize_vector_store(self, documents: List[Document] = None):
        """
        Initializes or loads the vector store.

        Args:
            documents (List[Document], optional): List of documents to initialize the vector store with.
        """
        if documents:
            self._batch_process_chroma_documents(documents)
        else:
            self.vector_stores["chroma"] = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        self.vs_initialized = True

    def create_retriever(
        self, llm, n_documents: int, bm25_portion: float = 0.8
    ) -> MultiQueryRetriever:
        """
        Creates a retriever using Chroma.

        Args:
            llm: Language model to use for the retriever.
            n_documents (int): Number of documents to retrieve.
            bm25_portion (float): Portion of BM25 to use in the retriever.

        Returns:
            MultiQueryRetriever: Configured retriever.
        """
        self.vector_store = MultiQueryRetriever.from_llm(
            retriever=self.vector_stores["chroma"].as_retriever(
                search_kwargs={"k": n_documents}
            ),
            llm=llm,
            include_original=True,
            prompt=DEFAULT_QUERY_PROMPT,
        )
        return self.vector_store

    def load_and_process_documents(self, doc_dir: str) -> List[Document]:
        """
        Loads and processes documents from the specified directory.

        Returns:
            List[Document]: List of processed documents.
        """
        loader = DocumentLoader(doc_dir)
        return loader.load_documents()
