"""
python -m src.llm_evaluation.run_llm_eval_predictions --input_folder data/clear_evaluation --output_folder data/llm_eval_predictions
"""

import json
import os
from glob import glob
from tqdm import tqdm
import argparse

from concurrent.futures import ThreadPoolExecutor
from ..rag_pipeline.rag_system import RAGSystem


docs_dir="data/docs"
persist_directory_dir="data/chroma_db"
batch_size=64

# Initialize RAG system
rag = RAGSystem(
    docs_dir,
    persist_directory_dir,
    batch_size
)
rag.initialize_vector_store()

def run_file_prediction(path: str, output_folder: str):
    raw: dict[str, str] = json.load(open(path))
    llm_answer = "".join(rag.query(raw["question"]))
    with open(os.path.join(output_folder, os.path.basename(path).replace("json", "txt")), "w") as file:
        file.write(llm_answer)
    return llm_answer

def run_predictions(input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    
    files = glob(os.path.join(input_folder, "*.json"))

    with ThreadPoolExecutor(max_workers=2) as thr:
        predictions = list(tqdm(thr.map(lambda x: run_file_prediction(x, output_folder), files), total=len(files)))
    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rag answers from text files.")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the folder containing input text files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder where output files will be saved.",
    )
    args = parser.parse_args()

    run_predictions(args.input_folder, args.output_folder)
