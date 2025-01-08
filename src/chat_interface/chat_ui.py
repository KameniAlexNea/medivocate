import logging
import os
from typing import Dict, List

import gradio as gr

logging.basicConfig(
    level=logging.ERROR,
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
        for text in self.rag_system.query(message, history):
            result += text
            yield result
        return result

    def create_interface(self):
        chat_interface = gr.ChatInterface(
            fn=self.respond,
            type="messages",
            title="Medivocate",
            description="Medivocate est une application qui offre des informations claires et structurées sur l'histoire de l'Afrique et sa médecine traditionnelle, en s'appuyant exclusivement sur un contexte issu de documentaires sur l'histoire du continent africain.",
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
    rag_system = RAGSystem(top_k_documents=12)
    rag_system.initialize_vector_store()

    chat_interface = ChatInterface(rag_system)
    chat_interface.launch(share=False)
