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
        summarize_prompt_template = """
        This is a text :

{paragraph}

Summarize it in less than 50 words and more than 10 words.
        """
        self.summarize_prompt = PromptTemplate(
            input_variables=["paragraph"], template=summarize_prompt_template
        )

        keywords_prompt_template = """Here is a paragraph:
{paragraph}

Extract the 3 most important keywords from this paragraph, separated by commas.
       """
        self.keywords_prompt = PromptTemplate(
            input_variables=["paragraph"], template=keywords_prompt_template
        )
        self.define_text_splitter()

    def define_text_splitter(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def clean_text(self, text: str):
        text = re.sub(r"\s+", " ", text)  # Remplace les espaces multiples par un seul
        text = re.sub(
            r"\n\s*\n+", "\n", text
        )  # Supprime les retours multiples à la ligne
        text = re.sub(r"[^\w\s\n.,;!?-]", "", text)
        return text.strip()

    def generate_summaries(self, paragraphs: str):
        summaries = []
        for paragraph in paragraphs:
            input_prompt = self.summarize_prompt.format(paragraph=paragraph)
            summary = self.llm(input_prompt)
            summaries.append(summary.strip())
        return summaries

    def generate_keywords(self, paragraphs: str):
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

    def retrieve_documents_from_file(self, file_path: str):
        with open(file_path, mode="r") as f:
            text = f.read()
        text = self.clean_text(text)
        large_chunks = self.split_text_into_large_chunks(text)
        summaries = self.generate_summaries(large_chunks)
        new_text = "\n".join(summaries)
        chunks = self.text_splitter.split_text(new_text)
        keywords_list = self.generate_keywords(chunks)
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

    chunkingManager = ChunkingManager()
