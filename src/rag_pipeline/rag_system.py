from typing import List, Optional

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from tqdm import tqdm

from ..utilities.llm_models import get_llm_model_chat, get_llm_model_embedding


class RAGSystem:
    def __init__(
        self,
        docs_dir: str = "data/docs",
        persist_directory_dir="data/chroma_db",
        batch_size: int = 64,
    ):
        self.docs_dir = docs_dir
        self.persist_directory_dir = persist_directory_dir
        self.batch_size = batch_size
        self.embeddings = self._get_embeddings()
        self.llm = self._get_llm()
        self.vector_store: Optional[Chroma] = None
        self.chain: Optional[Runnable] = None

    def _get_llm(self):
        return get_llm_model_chat("OLLAMA", temperature=0.1, max_tokens=256)

    def _get_embeddings(self):
        """Initialize embeddings based on environment configuration"""
        return get_llm_model_embedding()

    def load_documents(self) -> List:
        """Load and split documents from the specified directory"""
        loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        return splitter.split_documents(documents)

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

    def setup_rag_chain(self):
        if self.chain is not None:
            return
        """Set up the RAG chain with custom prompt"""
        prompt_template = """Using the context provided below, answer the question that follows as accurately as possible.

**Context**: 
{context}

**Question**: 
{input}

Answer (You should answer in the same language as the given question, even when you don't know):"""

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "input"]
        )
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        question_answer_chain = create_stuff_documents_chain(
            self.llm, prompt, output_parser=StrOutputParser()
        )

        self.chain = create_retrieval_chain(retriever, question_answer_chain)

    def query(self, question: str):
        """Query the RAG system"""
        if not self.vector_store:
            self.initialize_vector_store()

        self.setup_rag_chain()
        response = self.chain.invoke({"input": question})

        return {
            "answer": response["answer"],
            "source_documents": [doc.page_content for doc in response["context"]],
        }

    def query_iter(self, question: str):
        """Query the RAG system"""
        if not self.vector_store:
            self.initialize_vector_store()

        self.setup_rag_chain()
        for token in self.chain.stream({"input": question}):
            if "answer" in token:
                yield token["answer"]
