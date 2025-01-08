import logging
import os
from typing import List, Optional

from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import (
    ConversationalRetrievalChain,
)

from ..utilities.llm_models import get_llm_model_chat
from ..vector_store.vector_store import VectorStoreManager
from .prompts import CHAT_PROMPT


class RAGSystem:
    def __init__(
        self,
        docs_dir: str = "data/chunks",
        persist_directory_dir="data/chroma_db",
        batch_size: int = 64,
        top_k_documents=5,
    ):
        self.top_k_documents = top_k_documents
        self.llm = self._get_llm()
        self.chain: Optional[ConversationalRetrievalChain] = None
        self.vector_store_management = VectorStoreManager(
            docs_dir, persist_directory_dir, batch_size
        )

    def _get_llm(
        self,
    ):
        return get_llm_model_chat(temperature=0.1, max_tokens=500)

    def load_documents(self) -> List:
        """Load and split documents from the specified directory"""
        return self.vector_store_management.load_documents()

    def initialize_vector_store(self, documents: List = None):
        """Initialize or load the vector store"""
        self.vector_store_management.initialize_vector_store(documents)

    def setup_rag_chain(self):
        if self.chain is not None:
            return
        chain_type_kwargs = {"prompt": CHAT_PROMPT}
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        retriever = self.vector_store_management.vector_store.as_retriever(
            search_kwargs={"k": self.top_k_documents}
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            combine_docs_chain_kwargs=chain_type_kwargs,
            chain_type="stuff",
            memory=memory,
        )
        logging.info("RAG chain setup complete" + str(self.chain))

    def query(self, question: str):
        """Query the RAG system"""
        if not self.vector_store_management.vector_store:
            self.initialize_vector_store()

        self.setup_rag_chain()

        for token in self.chain.stream({"question": question}):
            if "answer" in token:
                yield token["answer"]


if __name__ == "__main__":
    from glob import glob

    docs_dir = "data/docs"
    persist_directory_dir = "data/chroma_db"
    batch_size = 64

    # Initialize RAG system
    rag = RAGSystem(docs_dir, persist_directory_dir, batch_size)

    if len(glob(os.path.join(persist_directory_dir, "*/*.bin"))):
        rag.initialize_vector_store()  # vector store initialized
    else:
        # Load and index documents
        documents = rag.load_documents()
        rag.initialize_vector_store(documents)  # documents

    print(rag.query("Quand a eu lieu la traite négrière ?"))
