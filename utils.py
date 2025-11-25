import os
from pinecone import Pinecone  # ← NEW: Import the Pinecone class
print("Success! Pinecone v5+ and langchain_openai are working")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone client (NEW v5+ way)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Note: No 'environment' needed anymore—it's handled via the API key (project-based)

index_name = "youtube-transcripts"

# Connect to your index
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Pinecone index '{index_name}' does not exist.")

index = pc.Index(index_name)  # ← Now use the client to get the index

# OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize chat model
chat_model = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def init_qa_chain():
    """Initialize a RetrievalQA chain using Pinecone and OpenAI embeddings."""
    from langchain_pinecone import PineconeVectorStore  # ← Use langchain_pinecone's wrapper (modern way)

    # Create vectorstore using langchain_pinecone (handles the client/index for you)
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings  # ← Pass the full embeddings object
    )

    qa = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4})  # ← Optional: Tune retrieval (top 4 docs)
    )
    return qa

def get_agent(tools: list):
    """Initialize an agent with given tools."""
    agent = initialize_agent(
        tools=tools,
        llm=chat_model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
    return agent


def get_agent(tools: list):
    """Initialize an agent with given tools."""
    agent = initialize_agent(
        tools=tools,
        llm=chat_model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
    return agent
