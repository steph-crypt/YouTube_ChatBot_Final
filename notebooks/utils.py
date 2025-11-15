
# import os
# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.agents import initialize_agent, Tool, AgentType

# def get_api_keys():
#     load_dotenv()
#     return {
#         "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
#         "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
#     }

# def init_qa_chain(index_name="youtube-transcripts"):
#     keys = get_api_keys()
#     embedding = OpenAIEmbeddings(openai_api_key=keys["OPENAI_API_KEY"])
#     pc = Pinecone(api_key=keys["PINECONE_API_KEY"])
#     index = pc.Index(index_name)
#     vectordb = PineconeVectorStore(index, embedding)
#     retriever = vectordb.as_retriever(search_kwargs={"k": 5})

#     llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=keys["OPENAI_API_KEY"])

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True
#     )
#     return qa_chain

# def get_agent(qa_chain):
#     keys = get_api_keys()

#     def answer_with_sources(input_text: str):
#         result = qa_chain({"query": input_text})
#         answer = result["result"]
#         sources = list(set(
#             doc.metadata.get("source_file", "Unknown") for doc in result.get("source_documents", [])
#         ))
#         if sources:
#             return f"{answer}\n\nSources:\n" + "\n".join(sources)
#         return answer

#     tools = [
#         Tool(
#             name="YouTubeTranscriptQA",
#             func=answer_with_sources,
#             description="Answer questions about YouTube transcripts"
#         )
#     ]
#     memory = ConversationBufferMemory(memory_key="chat_history")
#     llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=keys["OPENAI_API_KEY"])

#     return initialize_agent(
#         agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#         tools=tools,
#         llm=llm,
#         memory=memory,
#         verbose=True
#     )

# utils.py

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from pinecone import Pinecone
import os


# -----------------------------
# 🔑 Load Keys
# -----------------------------
def load_keys():
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY")
    }


# -----------------------------
# 📚 Initialize Retrieval QA Chain
# -----------------------------
def init_qa_chain(index_name="youtube-transcripts"):
    keys = load_keys()

    if not keys["OPENAI_API_KEY"]:
        raise ValueError("❌ Missing OPENAI_API_KEY in environment.")

    if not keys["PINECONE_API_KEY"]:
        raise ValueError("❌ Missing PINECONE_API_KEY in environment.")

    # Embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=keys["OPENAI_API_KEY"])

    # Connect to Pinecone
    pc = Pinecone(api_key=keys["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    # Vector store
    vectordb = PineconeVectorStore(index, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=keys["OPENAI_API_KEY"],
    )

    # RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa


# -----------------------------
# 🤖 Build Agent Wrapper
# -----------------------------
def get_agent(qa_chain):
    """Returns the main conversational agent that Gradio will call."""

    def answer_with_sources(query: str):
        response = qa_chain({"query": query})
        answer = response["result"]
        source_docs = response["source_documents"]

        sources = list({
            d.metadata.get("source_file", "Unknown")
            for d in source_docs
        })

        source_text = "\n".join(sources)
        return f"{answer}\n\nSources:\n{source_text}"

    tools = [
        Tool(
            name="YouTubeTranscriptQA",
            func=answer_with_sources,
            description="Answer questions using YouTube transcript RAG system."
        )
    ]

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        verbose=True,
        memory=memory,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        max_iterations=3
    )

    return agent
