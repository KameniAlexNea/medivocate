import os
from typing import List, Optional

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from utilities.llm_models import get_llm_model_chat, get_llm_model_embedding
from vector_store.vector_store import VectorStoreManager


class RAGSystem:
    def __init__(
        self,
        docs_dir: str = "data/docs",
        persist_directory_dir="data/chroma_db",
        batch_size: int = 64,
        top_k_documents=5,
    ):
        self.top_k_documents = top_k_documents
        self.embeddings = self._get_embeddings()
        self.llm = self._get_llm()
        self.chain: Optional[Runnable] = None
        self.vector_store_management = VectorStoreManager(
            docs_dir, persist_directory_dir, batch_size
        )

    def _get_llm(self):
        return get_llm_model_chat("OLLAMA", temperature=0.1, max_tokens=150)

    def _get_embeddings(self):
        """Initialize embeddings based on environment configuration"""
        return get_llm_model_embedding()

    def load_documents(self) -> List:
        """Load and split documents from the specified directory"""
        return self.vector_store_management.load_documents()

    def initialize_vector_store(self, documents: List = None):
        """Initialize or load the vector store"""
        self.vector_store_management.initialize_vector_store(documents)

    def setup_rag_chain(self):
        if self.chain is not None:
            return
        """Set up the RAG chain with custom prompt"""
        prompt_template = """Using the context provided below, answer the question that follows as accurately as possible.
If the answer cannot be determined from the context, respond with "I don't know." Avoid making up information.

**Context**: 
{context}

**Question**: 
{input}

Answer (You should answer in the same language as the given question):"""

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "input"]
        )
        retriever = self.vector_store_management.vector_store.as_retriever(
            search_kwargs={"k": self.top_k_documents}
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        self.chain = create_retrieval_chain(retriever, question_answer_chain)

    def query(self, question: str):
        """Query the RAG system"""
        if not self.vector_store_management.vector_store:
            self.initialize_vector_store()

        self.setup_rag_chain()
        response = self.chain.invoke({"input": question})

        return {
            "answer": response["answer"],
            "source_documents": [doc.page_content for doc in response["context"]],
        }

    def query_iter(self, question: str):
        """Query the RAG system"""
        if not self.vector_store_management.vector_store:
            self.initialize_vector_store()

        self.setup_rag_chain()
        for token in self.chain.stream({"input": question}):
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
