import json
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from tqdm import tqdm

from ..utilities.llm_models import get_llm_model_embedding


def sanitize_metadata(metadata):
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            # Convert lists to comma-separated strings or handle appropriately
            sanitized[key] = ", ".join(value)
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            raise ValueError(
                f"Unsupported metadata type for key '{key}': {type(value)}"
            )
    return sanitized


class VectorStoreManager:
    def __init__(self, docs_dir: str, persist_directory_dir: str, batch_size=64):
        self.embeddings = get_llm_model_embedding()
        self.vector_store = None
        self.docs_dir = docs_dir
        self.persist_directory_dir = persist_directory_dir
        self.batch_size = batch_size
        self.collection_name = os.getenv("OLLAM_EMB").split(":")[0]

    def _batch_process_documents(self, documents: List):
        """Process documents in batches"""
        for i in tqdm(
            range(0, len(documents), self.batch_size), desc="Processing documents"
        ):
            batch = documents[i : i + self.batch_size]

            if not self.vector_store:
                # Initialize vector store with first batch
                self.vector_store = Chroma.from_documents(
                    collection_name=self.collection_name,
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
                collection_name=self.collection_name,
                persist_directory=self.persist_directory_dir,
                embedding_function=self.embeddings,
            )

    def _load_text_documents(self) -> List:
        """*
        Load and split documents from the specified directory
        @TODO Move this function to chunking
        """
        loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        return splitter.split_documents(documents)

    def _load_json_documents(self) -> List:
        """*
        Load and split documents from the specified directory
        @TODO Move this function to chunking
        """
        files = glob(os.path.join(self.docs_dir, "*.json"))

        def load_json_file(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)["kwargs"]
            return Document.model_validate(
                {**data, "metadata": sanitize_metadata(data["metadata"])}
            )

        with ThreadPoolExecutor() as executor:
            documents = list(
                tqdm(
                    executor.map(load_json_file, files),
                    total=len(files),
                    desc="Loading JSON documents",
                )
            )

        return documents
    
    def load_documents(self) -> List:
        files = glob(os.path.join(self.docs_dir, "*.json"))
        if len(files):
            return self._load_json_documents()
        return self._load_text_documents()
