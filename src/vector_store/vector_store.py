import os
from typing import List

from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings, Chroma


class VectorStoreManager:
    def __init__(self, persist_directory_dir, batch_size=10):
        self.embeddings = self._get_embeddings()
        self.vector_store = None
        self.persist_directory_dir = persist_directory_dir
        self.batch_size = batch_size

    def _get_embeddings(self):
        """Initialize embeddings based on environment configuration"""
        return OllamaEmbeddings(
            model=os.getenv("OLLAM_EMB"),
            base_url=os.getenv("OLLAMA_HOST"),
            client_kwargs={
                "headers": {"Authorization": "Bearer " + os.getenv("OLLAMA_TOKEN")}
            },
        )
    
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