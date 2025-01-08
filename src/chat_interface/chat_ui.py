import logging
import os
from typing import Dict, List

import gradio as gr

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from ..rag_pipeline.rag_system import RAGSystem

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ChatInterface:
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.chat_history: List[Dict] = []

    def respond(self, message: str, history: List[List[str]]):
        result = ""
        logging.info(f"Received message: {message}")
        for text in self.rag_system.query(message):
            result += text
            yield result
        return result

    def create_interface(self):
        chat_interface = gr.ChatInterface(
            fn=self.respond,
            type="messages",
            title="Medivocate",
            description="Medivocate is an AI-driven platform leveraging Retrieval-Augmented Generation (RAG) powered by African history. It processes and classifies document pages with precision to provide trustworthy, personalized guidance, fostering accurate knowledge and equitable access to historical insights.",
            # retry_btn=None,
            # undo_btn=None,
            # clear_btn="Clear",
            # chatbot=gr.Chatbot(show_copy_button=True),
        )
        return chat_interface

    def launch(self, share=False):
        interface = self.create_interface()
        interface.launch(share=share)


# Usage example:
if __name__ == "__main__":
    rag_system = RAGSystem(top_k_documents=12, model_type="ollama")
    rag_system.initialize_vector_store()

    chat_interface = ChatInterface(rag_system)
    chat_interface.launch(share=False)
