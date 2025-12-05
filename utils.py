# utils.py
import os
from pinecone import Pinecone
print("Success! Pinecone v5+ and langchain_openai are working")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool                     # ← NEW: proper tool decorator
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# ==================== Pinecone Setup ====================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "youtube-transcripts"

if index_name not in pc.list_indexes().names():
    raise ValueError(f"Pinecone index '{index_name}' does not exist.")

index = pc.Index(index_name)

# ==================== Models & Embeddings ====================
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

chat_model = ChatOpenAI(
    model="gpt-4o-mini",           # or "gpt-4-turbo" if you prefer
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ==================== Vector Store + Retriever ====================
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"                # change to "content" if that's your field name
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ==================== Pure LCEL RAG Chain ====================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about YouTube video transcripts.

Use ONLY the following retrieved context to answer. 
If you don't know, just say:
"I don't know based on the available transcripts."

Context:
{context}

Question: {question}
Answer:""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | chat_model
    | StrOutputParser()
)

# ==================== The ONE Perfect Tool (with docstring) ====================
@tool
def youtube_transcript_qa(question: str) -> str:
    """Answer any question about the content of YouTube videos stored as transcripts in Pinecone.
    
    Use this tool for ALL questions related to video topics, guests, quotes, summaries,
    timestamps, opinions, explanations, or anything said in the videos.
    
    Examples:
    - "What did Elon Musk say about Mars?"
    - "Summarize the discussion on AI safety"
    - "Who was the guest in the last Lex Fridman podcast?"
    """
    answer = rag_chain.invoke(question)
    return answer

# ==================== Create Conversational Agent with Memory ====================
def get_agent():
    """Returns a fully configured ReAct agent with conversation memory and one powerful RAG tool."""
    memory = MemorySaver()

    agent_executor = create_react_agent(
        model=chat_model,
        tools=[youtube_transcript_qa],
        checkpointer=memory,
    )

    return agent_executor

# ==================== Optional: Direct stateless QA (if you ever want to bypass the agent) ====================
def rag_chain_with_sources(question: str):
    """Same as the tool but also returns source documents (useful for debugging or citations)."""
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    return {"answer": answer, "source_documents": docs}

# Keep this for backward compatibility if you use it elsewhere
get_qa_chain = lambda: rag_chain_with_sources