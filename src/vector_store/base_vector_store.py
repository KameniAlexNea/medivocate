"""Base vector store manager with common functionality."""
from abc import ABC, abstractmethod
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from ..config import VectorStoreConfig
from ..utilities.llm_models import get_llm_model_embedding


class BaseVectorStoreManager(ABC):
    """
    Base class for vector store managers with common functionality.
    """

    def __init__(self, persist_directory: str, batch_size: int = 64):
        """
        Initializes the base VectorStoreManager with common parameters.

        Args:
            persist_directory (str): Directory to persist the vector store.
            batch_size (int): Number of documents to process in each batch.
        """
        config = VectorStoreConfig(
            persist_directory=persist_directory, batch_size=batch_size
        )
        self.persist_directory = config.persist_directory
        self.batch_size = config.batch_size
        self.embeddings = get_llm_model_embedding()
        self.collection_name = config.collection_name
        self.vs_initialized = False

    def _batch_process_chroma_documents(self, documents: List[Document]):
        """
        Processes documents in batches for Chroma vector store initialization.
        This is the common Chroma processing logic shared by all subclasses.

        Args:
            documents (List[Document]): List of documents to process.
        """
        for i in tqdm(
            range(0, len(documents), self.batch_size), desc="Processing documents"
        ):
            batch = documents[i : i + self.batch_size]

            if not self.vs_initialized:
                self._initialize_chroma_store(batch)
                self.vs_initialized = True
            else:
                self._add_chroma_documents(batch)

    @abstractmethod
    def _initialize_chroma_store(self, documents: List[Document]):
        """
        Initialize the Chroma vector store with the first batch of documents.
        Must be implemented by subclasses.

        Args:
            documents (List[Document]): First batch of documents.
        """
        pass

    @abstractmethod
    def _add_chroma_documents(self, documents: List[Document]):
        """
        Add additional documents to the existing Chroma vector store.
        Must be implemented by subclasses.

        Args:
            documents (List[Document]): Documents to add.
        """
        pass

    @abstractmethod
    def initialize_vector_store(self, documents: List[Document] = None):
        """
        Initialize or load the vector store.
        Must be implemented by subclasses.

        Args:
            documents (List[Document], optional): Documents to initialize with.
        """
        pass

    @abstractmethod
    def create_retriever(self, llm, n_documents: int, bm25_portion: float = 0.8):
        """
        Create and return the appropriate retriever.
        Must be implemented by subclasses.

        Args:
            llm: Language model for the retriever.
            n_documents (int): Number of documents to retrieve.
            bm25_portion (float): Portion for BM25 (if applicable).

        Returns:
            The configured retriever.
        """
        pass