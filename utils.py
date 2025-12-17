import os
from pinecone import Pinecone
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "youtube-transcripts"

if index_name not in pc.list_indexes().names():
    raise ValueError(f"Pinecone index '{index_name}' does not exist.")

# Embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Chat model
chat_model = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Memory (optional)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="result"  # 🔥 THIS FIXES EVERYTHING
)


def init_qa_chain():
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    qa = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,  # optional
        return_source_documents=True
    )

    return qa
