


from chunking.chunk import get_chunks, get_keywords
from sentence_transformers import SentenceTransformer

def store_embeddings(file_path, embedding_version, chunk_size, chunk_overlap):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = get_chunks(text, chunk_size, chunk_overlap)
    keywords = get_keywords(chunks, embedding_version)

    embedding_model = SentenceTransformer(embedding_version)
    embeddings = [embedding_model.encode(chunk) for chunk in chunks]

    metadata = [
        {"chunk_index": i, "keywords": keywords[i], "source_file":file_path}
    for i in range(len(chunks))
    ]

    for i in range(min(10, len(chunks))):
        print(f"Chunk {i+1}: {chunks[i]}")
        print(f"Keywords: {metadata[i]['keywords']}")