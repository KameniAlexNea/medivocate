from langchain.text_splitter import RecursiveCharacterTextSplitter
from keybert import KeyBERT


def get_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)

    return chunks

def get_keywords(chunks, embedding_version):
    keybert_model = KeyBERT(embedding_version)

    keywords = [
        keybert_model.extract_keywords(chunk, top_n=3)
        for chunk in chunks
    ]

    keywords = [[kw[0] for kw in kw_list] for kw_list in keywords]

    return keywords
