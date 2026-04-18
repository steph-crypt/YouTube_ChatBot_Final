#!/usr/bin/env python3
"""
Rebuild the Pinecone index from datasets/transcripts/*.txt using the local HuggingFace
embedding model (same as embedding_config.get_embeddings()). No OpenAI required.

Usage (from project root):
  python scripts/reindex_pinecone.py

Requires: PINECONE_API_KEY, and enough disk/RAM for sentence-transformers (first run downloads weights).

This DELETES the existing index named PINECONE_INDEX_NAME (default: youtube-transcripts) if present,
then recreates it with the correct dimension for the chosen embedding model.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding_config import DEFAULT_EMBEDDING_MODEL, embedding_dimension, get_embeddings  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


def load_transcript_documents(transcripts_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(transcripts_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < 20:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source_file": path.name},
            )
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-index YouTube transcripts into Pinecone")
    parser.add_argument(
        "--index-name",
        default=os.getenv("PINECONE_INDEX_NAME", "youtube-transcripts"),
        help="Pinecone index name (default: youtube-transcripts)",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "transcripts",
        help="Directory containing .txt transcript files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Character chunk size for splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Documents per upsert batch",
    )
    parser.add_argument(
        "--skip-delete",
        action="store_true",
        help="Do not delete existing index (fails if dimension mismatch)",
    )
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY is not set in .env", file=sys.stderr)
        sys.exit(1)

    if not args.transcripts_dir.is_dir():
        print(f"ERROR: transcripts directory not found: {args.transcripts_dir}", file=sys.stderr)
        sys.exit(1)

    print("Loading embedding model (may download on first run)…")
    embeddings = get_embeddings()
    dim = embedding_dimension(embeddings)
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    print(f"  model={model_name!r}, dimension={dim}")

    print(f"Loading transcripts from {args.transcripts_dir}…")
    documents = load_transcript_documents(args.transcripts_dir)
    if not documents:
        print("ERROR: No .txt transcripts found.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(documents)} transcript file(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    split_docs = splitter.split_documents(documents)
    print(f"  {len(split_docs)} chunk(s) after splitting")

    pc = Pinecone(api_key=api_key)
    index_name = args.index_name

    existing = pc.list_indexes().names()
    if index_name in existing:
        if args.skip_delete:
            print(f"ERROR: Index {index_name!r} exists; re-run without --skip-delete to replace.", file=sys.stderr)
            sys.exit(1)
        print(f"Deleting existing index {index_name!r}…")
        pc.delete_index(index_name)

    print(f"Creating index {index_name!r} (dim={dim}, cosine, serverless aws us-east-1)…")
    pc.create_index(
        name=index_name,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    print("Uploading vectors…")
    for i in tqdm(range(0, len(split_docs), args.batch_size)):
        batch = split_docs[i : i + args.batch_size]
        vector_store.add_documents(batch)

    print(f"Done. Index {index_name!r} now has {len(split_docs)} vectors.")
    print("Set PINECONE_INDEX_NAME in .env if you used a non-default name.")


if __name__ == "__main__":
    main()
