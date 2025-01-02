import re
from uuid import uuid4

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import ChatOllama


class ChunkingManager:
    def __init__(self, chunk_size, chunk_overlap, llm: ChatOllama, nb_keywords=3):
        """summarize_threshold is the threshold from which the text is summarized."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.nb_keywords = nb_keywords
        self.llm = llm
        
        self.init_summarize_prompt()
        self.init_keywords_prompt()
        self.init_cleaner_prompt()
        
        self.define_text_splitter()

    def init_keywords_prompt(self):
        keywords_prompt_template = """Here is a text extracted from a book:
        {paragraph}

        Extract the 3 most important keywords from this text, separated by commas.
        """
        self.keywords_prompt = PromptTemplate(
            input_variables=["paragraph"], template=keywords_prompt_template
        )

    def init_summarize_prompt(self):
        summarize_prompt_template = """
        You are a summarization assistant. Your task is to summarize the following excerpt from a book while preserving the logical flow of ideas and the paragraph structure. 

        For each paragraph in the excerpt:
        1. Identify the main idea or theme discussed.
        2. Rewrite the paragraph in a concise manner while retaining the key points and relationships between concepts.
        3. Ensure the summarized text maintains the logical order and coherence of the original text.

        Here is the excerpt:

        {paragraph}

        Summarize the excerpt following the above instructions.

        """
        self.summarize_prompt = PromptTemplate(
            input_variables=["paragraph"], template=summarize_prompt_template
        )


    def define_text_splitter(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def init_cleaner_prompt(self):
        template="""
        You are a text cleaning assistant. Your task is to:
        1. Correct spelling mistakes.
        2. Split concatenated words while keeping the proper context.
        3. Remove any unnecessary characters or symbols that may degrade the performance of a retrieval-augmented generation (RAG) model.

        Input text:
        {text}

        Clean and corrected text:
        """

        self.cleaner_prompt = PromptTemplate(
            input_variables=["text"], template=template
        )

    def clean_text(self, text):
        prompt = self.cleaner_prompt.format(text=text)
        cleaned = self.llm(prompt)

        return cleaned

    def init_summaries(self, paragraphs: str):
        summaries = []
        for paragraph in paragraphs:
            input_prompt = self.summarize_prompt.format(paragraph=paragraph)
            summary = self.llm(input_prompt)
            summaries.append(summary.strip())
        return summaries

    def init_keywords(self, paragraphs: str):
        keywords_list = []
        for paragraph in paragraphs:
            input_prompt = self.keywords_prompt.format(paragraph=paragraph)
            keywords = self.llm(input_prompt)
            keywords_list.append(keywords.strip().split(","))
        return keywords_list

    def split_text_into_large_chunks(self, text: str, target_word_count=300):
        """
        on splitte le texte en chunks qui vont être résumés par la suite
        """
        paragraphs = [para.strip() for para in text.split("\n") if para.strip()]
        chunks = []
        current_chunk = []
        current_word_count = 0
        for paragraph in paragraphs:
            word_count = len(re.findall(r"\w+", paragraph))
            if current_word_count + word_count >= target_word_count:
                current_chunk.append(paragraph)
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            current_chunk.append(paragraph)
            current_word_count += word_count

        # Ajouter le dernier chunk s'il reste des paragraphes
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks

    def retrieve_documents_from_file(self, file_path: str, verbose=False):
        with open(file_path, mode="r") as f:
            text = f.read()
        text = self.clean_text(text)
        if verbose:
            print("Cleaned text:")
            print(text)
            print("***********************************\n")

        
        large_chunks = self.split_text_into_large_chunks(text)
        summaries = self.init_summaries(large_chunks)

        if verbose:
            for (chunk, summary) in zip(large_chunks, summaries):
                print('Text: ', chunk)
                print('Summary: ', summary)
                print("---------------------------------------")

        new_text = "\n".join(summaries)
        chunks = self.text_splitter.split_text(new_text)
        keywords_list = self.init_keywords(chunks)

        if verbose:
            print("Final chunks and their keywords: ")
            for (chunk, keywords) in zip(chunks, keywords_list):
                print("Chunk: ", chunk)
                print("Keywords: ", keywords)
                print("----------------------------------------")

        documents = []
        uuids = [str(uuid4()) for _ in range(len(chunks))]
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
    from ..utilities.llm_models import get_llm_model_chat
    llm = get_llm_model_chat()
    chunkingManager = ChunkingManager(chunk_size=300, chunk_overlap=50, llm=llm)

    file_path = "../../data/chunking_data_sample/Le Livre Des Morts Des Anciens egyptiens-page_0009.text"
    documents = chunkingManager.retrieve_documents_from_file(file_path=file_path, verbose=True)
