"""
Central config, loaded once from environment variables.
Keeping this in one place means every other module just does
`from src.config import settings` instead of scattering os.getenv() calls.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))
    collection_name: str = "osha_1910"


settings = Settings()
