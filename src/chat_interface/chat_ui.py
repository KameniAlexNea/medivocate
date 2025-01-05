from typing import Dict, List

import gradio as gr

from ..rag_pipeline.rag_system import RAGSystem


class ChatInterface:
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.chat_history: List[Dict] = []

    def respond(self, message: str, history: List[List[str]]):
        response = self.rag_system.query(message)
        sources = "\n\nSources:\n" + "\n".join(
            f"- {source[:500]}..." for source in response["source_documents"]
        )
        return response["answer"] + sources

    def create_interface(self):
        chat_interface = gr.ChatInterface(
            fn=self.respond,
            title="Medivocate",
            description="Medivocate is an AI-driven platform leveraging Retrieval-Augmented Generation (RAG) powered by African history. It processes and classifies document pages with precision to provide trustworthy, personalized guidance, fostering accurate knowledge and equitable access to historical insights.",
            retry_btn=None,
            undo_btn=None,
            clear_btn="Clear",
            chatbot=gr.Chatbot(show_copy_button=True)
        )
        return chat_interface

    def launch(self, share=False):
        interface = self.create_interface()
        interface.launch(share=share)


# Usage example:
if __name__ == "__main__":
    rag_system = RAGSystem()
    documents = None  # rag_system.load_documents()
    rag_system.initialize_vector_store(documents)

    chat_interface = ChatInterface(rag_system)
    chat_interface.launch()
