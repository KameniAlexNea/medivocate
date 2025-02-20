import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Union
from uuid import uuid4

from dotenv import load_dotenv
from keybert import KeyBERT
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import tiktoken
from ..preprocessing.processor import Processor
from ..utilities.llm_models import get_llm_model_chat
from .agents import CategoryAgent, CleanAgent, KeyWordAgent, SummaryAgent


class ChunkingManager:
    def __init__(
        self,
        llm: ChatOllama,
        chunk_size=512,
        chunk_overlap=75,
        top_n=3,
        keyphrase_ngram_range=(1, 1),
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm
        self.top_n = top_n
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.kwb = KeyBERT(SentenceTransformer("all-mpnet-base-v2", device="cuda:0"))

        self.summary_agent = SummaryAgent(llm)
        self.clean_agent = CleanAgent(llm)
        self.keyword_agent = KeyWordAgent(llm)
        self.category_agent = CategoryAgent(llm)
        self.processor = Processor(chunk_size, chunk_overlap)

    def clean_text(self, text):
        return self.clean_agent.process(text)

    def generate_summaries(self, paragraphs):
        return self.summary_agent.batch_process(paragraphs)

    def generate_keywords(self, paragraphs: Union[str, list[str]], use_llm=True):
        if use_llm:
            return self.keyword_agent.batch_process(paragraphs)

        keywords = self.kwb.extract_keywords(
            paragraphs,
            top_n=self.top_n,
            keyphrase_ngram_range=self.keyphrase_ngram_range,
        )
        if isinstance(keywords[0], tuple):
            keywords = [keywords]
        keywords_list = [[kw[0] for kw in para_kw] for para_kw in keywords]
        return keywords_list

    def split_text_into_large_chunks(self, text, target_word_count=500):
        return self.processor.split_text_into_large_chunks(text, target_word_count)

    def retrieve_documents_from_folder(
        self,
        folder_path,
        use_llm_cleaning=False,
        use_llm_for_keywords=False,
        summarize_before_chunk=False,
        check_text_validity=True,
        llm_check_text_validity=False,
        verbose=False,
        target_word_count=500,
    ):
        """
        Load all .txt files in the folder (which represents a book), merge their contents,
        and process them similarly to a single file.
        """
        try:
            # Get all text files in the folder
            files = sorted(glob(os.path.join(folder_path, "*.txt")))
            if not files:
                logging.warning(f"No text files found in folder: {folder_path}")
                return []
            merged_text = ""
            for file in files:
                with open(file, mode="r", encoding="utf-8") as f:
                    merged_text += f.read() + "\n"

            if verbose:
                print(f"{'*' * 38}\nMerged text from folder {folder_path}:\n{merged_text}\n{'-' * 25}\n")

            text = merged_text

            if use_llm_cleaning:
                text = self.clean_text(text)

            if check_text_validity:
                if llm_check_text_validity:
                    category = self.category_agent.process(text)
                    if verbose:
                        print("---- Document Category -----")
                        print(category)
                    if ("contenu" not in category.lower() and not self.processor.is_valid_file(text)):
                        logging.warning(f"Invalid text in folder: {folder_path}")
                        return []

            if verbose:
                print(f"Cleaned text:\n{text}\n{'*' * 38}\n")

            if summarize_before_chunk:
                large_chunks = self.split_text_into_large_chunks(text, target_word_count)
                summaries = self.generate_summaries(large_chunks)
                if verbose:
                    print("****** Summary ******")
                    for chunk, summary in zip(large_chunks, summaries):
                        print(f"\nText: \n{chunk}\n\nSummary: \n{summary}\n")
                text = "\n".join(summaries)

            chunks = self.processor.text_splitter.split_text(text)
            keywords_list = self.generate_keywords(chunks, use_llm=use_llm_for_keywords)

            if verbose:
                print("****** Chunks and Keywords ******")
                for chunk, keywords in zip(chunks, keywords_list):
                    print(f"\nChunk: \n{chunk}\nKeywords: \n{keywords}\n")

            documents = [
                Document(
                    page_content=chunk,
                    metadata={"source": folder_path, "keywords": keywords, "chunk_index": str(i)},
                    id=str(uuid4().hex),
                )
                for i, (chunk, keywords) in enumerate(zip(chunks, keywords_list))
            ]
            return documents

        except Exception as e:
            logging.error(f"Error processing folder {folder_path}: {e}")
            return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate questions from OCR'd books.")
    parser.add_argument("--input_folder", type=str, help="Path to folder containing book sub-folders.")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size.")
    parser.add_argument("--chunk_overlap", type=int, default=75, help="Chunk overlap.")
    parser.add_argument("--save_folder", type=str, help="Path to save extracted chunks.")
    args = parser.parse_args()

    load_dotenv()

    llm = get_llm_model_chat(temperature=0.1, max_tokens=256)
    chunking_manager = ChunkingManager(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, llm=llm
    )

    os.makedirs(args.save_folder, exist_ok=True)

    # Each subfolder in input_folder is a book
    folders = sorted(glob(os.path.join(args.input_folder, "*")))
    folders = [i for i in folders if os.path.isdir(i)]

    def process_and_save(folder_path):
        documents = chunking_manager.retrieve_documents_from_folder(
            folder_path=folder_path,
            verbose=False,
            use_llm_for_keywords=False,
            summarize_before_chunk=False,
            check_text_validity=False,
            llm_check_text_validity=False,
            target_word_count=500,
        )

        if not documents:
            logging.warning(f"No documents extracted from {folder_path}")
        for doc in documents:
            with open(
                os.path.join(args.save_folder, f"{doc.id}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(doc.to_json(), f, indent=4, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_and_save, folders), total=len(folders)))
