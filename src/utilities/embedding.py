import os
from typing import List, Any
import logging
import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpointEmbeddings,
)
from pydantic import BaseModel, Field


class CustomEmbedding(BaseModel, Embeddings):
    hosted_embedding: HuggingFaceEndpointEmbeddings = Field(default_factory=lambda: None)
    cpu_embedding: HuggingFaceEmbeddings = Field(default_factory=lambda: None)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.hosted_embedding = HuggingFaceEndpointEmbeddings(
            model=os.getenv("HF_MODEL"),
            model_kwargs={"encode_kwargs": {"normalize_embeddings": True}},
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        self.cpu_embedding = HuggingFaceEmbeddings(
            model_name=os.getenv("HF_MODEL"),  # You can replace with any HF model
            model_kwargs={"device": "cpu" if not torch.cuda.is_available() else "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using the hosted embedding. If the API request limit is reached,
        fall back to using the CPU embedding.

        Args:
            texts (List[str]): List of documents to embed.

        Returns:
            List[List[float]]: List of embeddings for each document.
        """
        try:
            return self.hosted_embedding.embed_documents(texts)
        except:
            logging.warning("Issue with batch hosted embedding, moving to CPU")
            return self.cpu_embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query using the hosted embedding. If the API request limit is reached,
        fall back to using the CPU embedding.

        Args:
            text (str): Query to embed.

        Returns:
            List[float]: Embedding for the query.
        """
        try:
            return self.hosted_embedding.embed_query(text)
        except:
            logging.warning("Issue with hosted embedding, moving to CPU")
            return self.cpu_embedding.embed_query(text)
