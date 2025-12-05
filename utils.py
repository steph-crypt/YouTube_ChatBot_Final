# utils.py
import os
from pinecone import Pinecone
print("Success! Pinecone v5+ and langchain_openai are working")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
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
    model="gpt-4o-mini",  # or gpt-4-turbo, etc.
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ==================== Vector Store + Retriever ====================
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"  # adjust if your metadata field is different (e.g. "content")
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})  # top 6 chunks

# ==================== Memory ====================
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ==================== Pure LCEL RAG Chain (Modern Way) ====================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt template — uses {context} and {question} (standard for agents too)
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about YouTube video transcripts.

Use only the following retrieved context to answer the question.
If you don't know the answer, say "I don't know based on the available transcripts."

Context:
{context}

Question: {question}
Answer:""")

# Core RAG chain (no legacy create_* functions!)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | chat_model
    | StrOutputParser()
)

# Add source documents support (very useful!)
def rag_chain_with_sources(question: str):
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    return {
        "answer": answer,
        "source_documents": docs
    }

# Wrap as a Lambda so we can attach the helper
rag_chain_with_sources_func = RunnableLambda(rag_chain_with_sources)
rag_chain_with_sources_func.with_sources = rag_chain_with_sources  # optional alias

# ==================== Agent Tool ====================
qa_tool = Tool(
    name="YouTubeTranscriptQA",
    description="Useful for answering questions about YouTube video content and transcripts. Input must be a full question.",
    func=rag_chain_with_sources  # returns dict with answer + sources
)

# ==================== Initialize Agent with Memory ====================
def get_agent():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    agent = initialize_agent(
        tools=[qa_tool],
        llm=chat_model,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        max_iterations=4,
        handle_parsing_errors=True,
        early_stopping_method="generate"
    )
    return agent

# ==================== Optional: Direct QA Chain (if you ever want to skip agent) ====================
def get_qa_chain = lambda: rag_chain_with_sources_func