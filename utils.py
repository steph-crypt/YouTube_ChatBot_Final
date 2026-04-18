import os
from pathlib import Path

from pinecone import Pinecone
from dotenv import load_dotenv

# Load .env from project root (works even if you run from another cwd)
_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore

from embedding_config import get_embeddings

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX_NAME", "youtube-transcripts")

if index_name not in pc.list_indexes().names():
    raise ValueError(
        f"Pinecone index '{index_name}' does not exist. "
        f"Run: python scripts/reindex_pinecone.py"
    )

if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError(
        "ANTHROPIC_API_KEY is not set. Add it to your .env file (Anthropic Console → API keys)."
    )

# Local Hugging Face embeddings — must match scripts/reindex_pinecone.py
embeddings = get_embeddings()

# Claude answers; retrieval grounds answers in transcript chunks only
# Default matches current Claude API (3.5 snapshot IDs return 404 once retired)
_anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
chat_model = ChatAnthropic(
    model=_anthropic_model,
    temperature=0,
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# Strict transcript-only prompt (RAG)
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You answer using ONLY the YouTube transcript excerpts below. Do not rely on outside facts.

Rules:
- If the excerpts contain enough information, answer clearly and cite ideas from them.
- If the excerpts do not contain the answer, reply exactly: "The available transcripts do not contain that information."
- Do not invent quotes, episodes, or facts that are not supported by the excerpts.

Transcript excerpts:
{context}

Question: {question}

Answer:""",
)


def init_qa_chain():
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )

    # Stateless: no ConversationBufferMemory — memory caused repeated / stale answers per turn.
    qa = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    return qa


def get_agent(qa_chains):
    """Thin shim so `app.py` / notebooks can call `agent.run(question)`."""
    qa = qa_chains[0] if isinstance(qa_chains, (list, tuple)) else qa_chains

    class _AgentShim:
        def run(self, question: str) -> str:
            result = qa.invoke({"query": question})
            if isinstance(result, dict) and "result" in result:
                return result["result"]
            return str(result)

    return _AgentShim()
