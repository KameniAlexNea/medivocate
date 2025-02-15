import argparse
import os
from glob import glob
os.environ["IS_APP"] = "0"

from ..rag_pipeline.rag_system import RAGSystem

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create vector store for RAG system.")
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="data/docs",
        help="Directory containing documents.",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="data/chroma_db",
        help="Directory to persist vector store.",
    )
    parser.add_argument(
        "--batch_size", type=str, default=64, help="Embedding batch size."
    )
    args = parser.parse_args()

    docs_dir = args.docs_dir
    persist_directory_dir = args.persist_dir
    batch_size = args.batch_size

    # Initialize RAG system
    rag = RAGSystem(docs_dir, persist_directory_dir, batch_size)

    if len(glob(os.path.join(persist_directory_dir, "*/*.bin"))):
        rag.initialize_vector_store()  # vector store initialized
    else:
        # Load and index documents
        documents = rag.load_documents()
        print("Documents loaded", len(documents))

        rag.initialize_vector_store(documents)  # documents
