# utils.py
import os
from pinecone import Pinecone
print("Success! Pinecone v5+ and langchain_openai are working")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "youtube-transcripts"

if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' not found in your Pinecone account!")

index = pc.Index(index_name)

# Models
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Retriever
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template("""
You are an expert on YouTube video transcripts.
Answer using ONLY the context below. If unsure, say "I don't know."

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

# Tool
@tool
def youtube_transcript_qa(question: str) -> str:
    """Answer questions about YouTube video transcripts stored in Pinecone."""
    return rag_chain.invoke(question)

# Agent with memory
def get_agent():
    memory = MemorySaver()
    return create_react_agent(
        model=chat_model,
        tools=[youtube_transcript_qa],
        checkpointer=memory,
    )