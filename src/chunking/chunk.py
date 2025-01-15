import logging
from uuid import uuid4

from keybert import KeyBERT
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

from ..preprocessing.processor import Processor
from .agents import CategoryAgent, CleanAgent, KeyWordAgent, SummaryAgent


class ChunkingManager:
    def __init__(
        self,
        llm: ChatOllama,
        chunk_size=1000,
        chunk_overlap=200,
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

    def generate_summaries(self, paragraphs: list[str]):
        summaries = self.summary_agent.batch_process(paragraphs)
        return summaries

    def generate_keywords(self, paragraphs: list[str], use_llm=True):
        if use_llm:
            return self.keyword_agent.batch_process(paragraphs)
        keywords_list: list[str] = []
        for paragraph in paragraphs:
            keywords = self.kwb.extract_keywords(
                paragraph,
                top_n=self.top_n,
                keyphrase_ngram_range=self.keyphrase_ngram_range,
            )
            keywords = [kw[0] for kw in keywords]
            keywords_list.append(keywords)
        return keywords_list

    def split_text_into_large_chunks(self, text: str, target_word_count=500):
        """
        on splitte le texte en chunks qui vont être résumés par la suite
        """
        return self.processor.split_text_into_large_chunks(text, target_word_count)

    def retrieve_documents_from_file(
        self,
        file_path: str,
        use_llm_cleaning=False,
        use_llm_for_keywords=False,  # no use of llm for keywords, KeyBert is used instead
        summarize_before_chunk=False,  # no summarization before chunking
        check_text_validity=True,  # check text validity using llm ? simple approach
        llm_check_text_validity=False,  # check text validity using llm ? simple approach
        verbose=False,
        target_word_count=500,
    ):
        with open(file_path, mode="r") as f:
            text = f.read()

        if verbose:
            print(("*" * 38) + "\n")
            print("Raw text:\n")
            print(text)
            print(("-" * 25) + "\n")

        # text = self.processor.merge_sentences(text)

        if use_llm_cleaning:
            text = self.clean_text(text)

        if check_text_validity:
            checker = True
            if llm_check_text_validity:
                category = self.category_agent.process(text)
                if verbose:
                    print("---- Document Category -----")
                    print(category)

                checker = "contenu" not in category.lower()
            if not self.processor.is_valid_file(text) and checker:
                print(text)
                print("The text is invalid, not retrieving documents from it.")
                return []

        if verbose:
            print("Cleaned text:\n")
            print(text)
            print(("*" * 38) + "\n")

        if summarize_before_chunk:
            large_chunks = self.split_text_into_large_chunks(
                text, target_word_count=target_word_count
            )
            summaries = self.generate_summaries(large_chunks)

            if verbose:
                print("****** Summary ******")
                for chunk, summary in zip(large_chunks, summaries):
                    print("\n---------------------------------------")
                    print("Text: \n", chunk, "\n")
                    print("Summary: \n", summary, "\n")
                    print("---------------------------------------\n")

            text = "\n".join(summaries)

        chunks = self.processor.text_splitter.split_text(text)
        keywords_list = self.generate_keywords(chunks, use_llm=use_llm_for_keywords)

        if verbose:
            print("****** Chunks and Keywords ******")
            for chunk, keywords in zip(chunks, keywords_list):
                print("\n---------------------------------------")
                print("Chunk: \n", chunk)
                print("Keywords: \n", keywords)
                print("----------------------------------------\n")

        documents: list[Document] = []
        uuids = [str(uuid4().hex) for _ in range(len(chunks))]
        for id, chunk, keywords in zip(uuids, chunks, keywords_list):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": file_path, "keywords": keywords},
                    id=id,
                )
            )
        return documents


if __name__ == "__main__":
    import argparse
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor
    from glob import glob

    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Generate questions from text files.")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the folder containing input text files.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Path to the folder containing input text files.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Path to the folder containing input text files.",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        help="Path to save extracted chunks",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    from ..utilities.llm_models import get_llm_model_chat

    load_dotenv()

    llm = get_llm_model_chat(temperature=0.1, max_tokens=256)
    chunkingManager = ChunkingManager(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, llm=llm
    )

    os.makedirs(args.save_folder, exist_ok=True)

    files = sorted(glob(os.path.join(args.input_folder, "*/*.txt")))

    def get_and_save_document_chunking(file_path):
        try:
            documents = chunkingManager.retrieve_documents_from_file(
                file_path=file_path,
                verbose=False,
                use_llm_for_keywords=False,
                summarize_before_chunk=False,
                check_text_validity=False,
                llm_check_text_validity=False,
                target_word_count=500,
            )
            return documents, file_path
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            return [], file_path

    with ThreadPoolExecutor(max_workers=4) as executor:
        for documents, file_path in tqdm(
            executor.map(get_and_save_document_chunking, files), total=len(files)
        ):
            if not len(documents):
                logging.warning(f"No documents extracted from {file_path}")
            for doc in documents:
                with open(
                    os.path.join(args.save_folder, f"{doc.id}.json"),
                    mode="w",
                    encoding="utf-8",
                ) as f:
                    json.dump(doc.to_json(), f, indent=4)
