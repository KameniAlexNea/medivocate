import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from tqdm import tqdm


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
        self.vector_store = None
        self.chain = None

    def _get_llm(self):
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL"),
            temperature=0.1,
            max_tokens=255,
            # other params...
            base_url=os.getenv("OLLAMA_HOST"),
            client_kwargs={
                "headers": {"Authorization": "Bearer " + os.getenv("OLLAMA_TOKEN")}
            },
        )

    def _get_embeddings(self):
        """Initialize embeddings based on environment configuration"""
        return OllamaEmbeddings(
            model=os.getenv("OLLAM_EMB"),
            base_url=os.getenv("OLLAMA_HOST"),
            client_kwargs={
                "headers": {"Authorization": "Bearer " + os.getenv("OLLAMA_TOKEN")}
            },
        )

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
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {input}
        Answer:"""

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "input"]
        )
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 7})
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

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
