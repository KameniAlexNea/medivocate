import os
from typing import List

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

from ..utilities.llm_models import get_llm_model_embedding

class VectorStoreManager:
    def __init__(self, persist_directory_dir, batch_size=64):
        self.embeddings = get_llm_model_embedding()
        self.vector_store = None
        self.persist_directory_dir = persist_directory_dir
        self.batch_size = batch_size

    def _batch_process_documents(self, documents: List):
        """Process documents in batches"""
        for i in tqdm(
            range(0, len(documents), self.batch_size), desc="Processing documents"
        ):
            batch = documents[i : i + self.batch_size]

            if not self.vector_store:
                # Initialize vector store with first batch
                self.vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory_dir,
                )
            else:
                # Add subsequent batches
                self.vector_store.add_documents(batch)

    def initialize_vector_store(self, documents: List = None):
        """Initialize or load the vector store"""
        if documents:
            self._batch_process_documents(documents)
        else:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory_dir,
                embedding_function=self.embeddings,
            )
