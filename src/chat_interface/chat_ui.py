from typing import Dict, List

import gradio as gr

from ..rag_pipeline.rag_system import RAGSystem


class ChatInterface:
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.chat_history: List[Dict] = []

    def respond(self, message: str, history: List[List[str]]):
        text = ""
        for res in self.rag_system.query_iter(message):
            text += res
            yield text

    def create_interface(self):
        chat_interface = gr.ChatInterface(
            fn=self.respond,
            title="RAG Chat Assistant",
            description="Ask questions africa history. This is not a chat system, only a tool whare you can fact checking african history",
            retry_btn=None,
            undo_btn=None,
            clear_btn="Clear",
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
