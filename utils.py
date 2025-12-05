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

# ==================== Pinecone ====================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "youtube-transcripts"

if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' not found!")

index = pc.Index(index_name)

# ==================== Models ====================
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

chat_model = ChatOpenAI(
    model="gpt-4o-mini",          # or gpt-4-turbo
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ==================== Retriever ====================
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"               # change if your field is called "content"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ==================== RAG Chain ====================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template("""
You are an expert on YouTube video transcripts.
Answer ONLY using ONLY the context below. 
If you don't know, say "I don't know based on the available transcripts."

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

# ==================== Tool (with proper docstring) ====================
@tool
def youtube_transcript_qa(question: str) -> str:
    """Answer any question about YouTube video transcripts stored in the Pinecone index.
    
    Use this for questions about topics, guests, quotes, summaries, timestamps, etc.
    """
    return rag_chain.invoke(question)

# ==================== Agent with Memory ====================
def get_agent():
    memory = MemorySaver()
    agent = create_react_agent(
        model=chat_model,
        tools=[youtube_transcript_qa],
        checkpointer=memory,
    )
    return agent