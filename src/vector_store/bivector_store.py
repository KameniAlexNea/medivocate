import os
from typing import List, Union

from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from transformers import AutoTokenizer

from ..config import VectorStoreConfig
from .base_vector_store import BaseVectorStoreManager
from .prompts import DEFAULT_QUERY_PROMPT


class EnsembleVectorStoreManager(BaseVectorStoreManager):
    """
    Manages ensemble vector store (Chroma + BM25) initialization, updates, and retrieval.
    """

    def __init__(self, persist_directory: str, batch_size: int = 64):
        """
        Initializes the VectorStoreManager with the given parameters.

        Args:
            persist_directory (str): Directory to persist the vector store.
            batch_size (int): Number of documents to process in each batch.
        """
        super().__init__(persist_directory, batch_size)
        self.vector_stores: dict[str, Union[Chroma, BM25Retriever]] = {
            "chroma": None,
            "bm25": None,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.getenv("HF_MODEL", "meta-llama/Llama-3.2-1B")
        )
        self.vector_store = None

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

    def _batch_process_documents(self, documents: List[Document]):
        """
        Processes documents in batches for vector store initialization.
        Overrides base class to also initialize BM25 after Chroma processing.

        Args:
            documents (List[Document]): List of documents to process.
        """
        # Process Chroma documents using parent method
        self._batch_process_chroma_documents(documents)

        # Initialize BM25 with all documents
        self.vector_stores["bm25"] = BM25Retriever.from_documents(
            documents, tokenizer=self.tokenizer
        )

    def initialize_vector_store(self, documents: List[Document] = None):
        """
        Initializes or loads the vector store.

        Args:
            documents (List[Document], optional): List of documents to initialize the vector store. Defaults to None.
        """
        if documents:
            self._batch_process_documents(documents)
        else:
            self.vector_stores["chroma"] = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            all_documents = self.vector_stores["chroma"].get(
                include=["documents", "metadatas"]
            )
            documents = [
                Document(page_content=content, id=doc_id, metadata=metadata)
                for content, doc_id, metadata in zip(
                    all_documents["documents"],
                    all_documents["ids"],
                    all_documents["metadatas"],
                )
            ]
            self.vector_stores["bm25"] = BM25Retriever.from_documents(
                documents, tokenizer=self.tokenizer
            )
        self.vs_initialized = True

    def create_retriever(
        self, llm, n_documents: int, bm25_portion: float = 0.8
    ) -> EnsembleRetriever:
        """
        Creates an ensemble retriever combining Chroma and BM25.

        Args:
            llm: Language model to use for retrieval.
            n_documents (int): Number of documents to retrieve.
            bm25_portion (float): Proportion of BM25 retriever in the ensemble.

        Returns:
            EnsembleRetriever: The created ensemble retriever.
        """
        self.vector_stores["bm25"].k = n_documents
        self.vector_store = MultiQueryRetriever.from_llm(
            retriever=EnsembleRetriever(
                retrievers=[
                    self.vector_stores["bm25"],
                    self.vector_stores["chroma"].as_retriever(
                        search_kwargs={"k": n_documents}
                    ),
                ],
                weights=[bm25_portion, 1 - bm25_portion],
            ),
            llm=llm,
            include_original=True,
            prompt=DEFAULT_QUERY_PROMPT,
        )
        return self.vector_store
