import os
from pathlib import Path

from dotenv import load_dotenv

dotenv_path = Path(__file__).parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# Set defaults using values from .env
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
os.environ.setdefault("OLLAM_EMB", "nomic-embed-text")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434/")
os.environ.setdefault("OLLAMA_MODEL", "phi4")
os.environ.setdefault("OLLAMA_TOKEN", "")
os.environ.setdefault("USE_HF_EMBEDDING", "1")
os.environ.setdefault("USE_OLLAMA_CHAT", "0")
os.environ.setdefault("HF_MODEL", "nomic-ai/nomic-embed-text-v2-moe")
os.environ.setdefault("MAX_MESSAGES", "3")
os.environ.setdefault("N_CONTEXT", "3")
os.environ.setdefault("IS_APP", "0")
os.environ.setdefault("GDRIVE_URL", "https://drive.google.com/uc")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGSMITH_API_KEY", "")
os.environ.setdefault("LANGSMITH_PROJECT", "medivocate")
