from loguru import logger
import os
from typing import Generator, List, Optional

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.chains.history_aware_retriever import (
    create_history_aware_retriever,
)
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document

from ..config import RAGConfig
from ..utilities.llm_models import get_llm_model_chat
from ..vector_store.vector_store import VectorStoreManager
from .prompts import CHAT_PROMPT, CONTEXTUEL_QUERY_PROMPT


class RAGSystem:
    """Retrieval-Augmented Generation system for document Q&A.

    This class provides a complete RAG pipeline that:
    1. Loads and processes documents from a directory
    2. Creates or loads a vector store for document embeddings
    3. Sets up a conversational retrieval chain
    4. Provides streaming query capabilities with chat history

    Attributes:
        config: Configuration object containing all settings
        llm: Language model for generation
        chain: LangChain conversational retrieval chain
        vector_store_management: Vector store manager instance
        docs_dir: Directory containing documents to process
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the RAG system.

        Args:
            config: Optional configuration object. If None, uses environment variables or defaults.
        """
        self.config = config or RAGConfig.from_env()
        self._llm = None  # Lazy initialization
        self.chain: Optional[BaseConversationalRetrievalChain] = None
        self.vector_store_management = VectorStoreManager(
            self.config.persist_directory, self.config.batch_size
        )
        self.docs_dir = self.config.docs_dir

    @property
    def llm(self):
        """Get the language model with lazy initialization."""
        if self._llm is None:
            self._llm = self._get_llm()
        return self._llm

    def _get_llm(self):
        """Get the language model with configured parameters."""
        return get_llm_model_chat(
            temperature=self.config.temperature, max_tokens=self.config.max_tokens
        )

    def load_documents(self) -> List[Document]:
        """Load and split documents from the specified directory"""
        try:
            return self.vector_store_management.load_and_process_documents(
                self.docs_dir
            )
        except Exception as e:
            logger.error(f"Failed to load documents from {self.docs_dir}: {e}")
            raise

    def initialize_vector_store(
        self, documents: Optional[List[Document]] = None
    ) -> None:
        """Initialize or load the vector store"""
        try:
            self.vector_store_management.initialize_vector_store(documents)
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def setup_rag_chain(self) -> BaseConversationalRetrievalChain:
        """Setup the RAG chain with error handling."""
        if self.chain is not None:
            return self.chain

        try:
            retriever = self.vector_store_management.create_retriever(
                self.llm,
                self.config.top_k_documents,
                bm25_portion=self.config.bm25_portion,
            )

            # Contextualize question
            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, retriever, CONTEXTUEL_QUERY_PROMPT
            )
            self.question_answer_chain = create_stuff_documents_chain(
                self.llm, CHAT_PROMPT
            )
            self.chain = create_retrieval_chain(
                self.history_aware_retriever, self.question_answer_chain
            )
            logger.info("RAG chain setup complete")
            return self.chain
        except Exception as e:
            logger.error(f"Failed to setup RAG chain: {e}")
            raise

    def query(
        self, question: str, history: Optional[List[str]] = None
    ) -> Generator[str, None, None]:
        """Query the RAG system with streaming response.

        Args:
            question: The question to ask
            history: Optional chat history

        Yields:
            Chunks of the answer as they are generated
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if history is None:
            history = []

        try:
            if not self.vector_store_management.vs_initialized:
                self.initialize_vector_store()

            self.setup_rag_chain()

            for token in self.chain.stream(
                {"input": question, "chat_history": history}
            ):
                if "answer" in token:
                    yield token["answer"]
        except Exception as e:
            logger.error(f"Failed to query RAG system: {e}")
            raise

    def query_complex(self, question: str, verbose: bool = False) -> None:
        """Complex query method - not implemented yet"""
        raise NotImplementedError("Complex query method not yet implemented")


if __name__ == "__main__":
    from glob import glob

    from dotenv import load_dotenv

    # loading variables from .env file
    load_dotenv()

    # Use configuration from environment or defaults
    config = RAGConfig.from_env()

    # Initialize RAG system
    rag = RAGSystem(config)

    if len(glob(os.path.join(config.persist_directory, "*/*.bin"))):
        rag.initialize_vector_store()  # vector store initialized
    else:
        # Load and index documents
        documents = rag.load_documents()
        rag.initialize_vector_store(documents)  # documents

    queries = [
        "Quand a eu lieu la traite négrière ?",
        "Explique moi comment soigner la tiphoide puis le paludisme",
        "Quels étaient les premiers peuples d'afrique centrale et quelles ont été leurs migrations?",
    ]

    print("RAG System Demo")

    for query in queries:
        print(f"Query: {query}")
        print("Answer:")
        for chunk in rag.query(question=query):
            print(chunk, end="")
        print("\n" + "=" * 50 + "\n")
