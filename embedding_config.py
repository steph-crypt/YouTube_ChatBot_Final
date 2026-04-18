"""
Shared local embedding model (no OpenAI). Must match between re-indexing and runtime queries.
Default: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

# Override with EMBEDDING_MODEL in .env if you re-index with another HF model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_embeddings():
    """Same HuggingFace embedding model for Pinecone ingest and query."""
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": _pick_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


def embedding_dimension(embeddings) -> int:
    return len(embeddings.embed_query("dimension probe"))
