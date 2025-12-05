# utils.py
import os
from pinecone import Pinecone
print("Success! Pinecone v5+ and langchain_openai are working")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # Modern memory
from langgraph.prebuilt import create_react_agent  # Modern agent builder
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
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ==================== Vector Store + Retriever ====================
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"  # Adjust if needed
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ==================== Pure LCEL RAG Chain ====================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about YouTube video transcripts.

Use only the following retrieved context. 
If you don't know the answer, just say "I don't know based on the available transcripts."

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

def rag_chain_with_sources(question: str):
    """Answer a question about YouTube transcripts using RAG retrieval from Pinecone.
    
    This tool retrieves relevant transcript chunks, generates an answer based on them,
    and returns the answer along with source documents for citation.
    
    Args:
        question: A natural language question about video content (e.g., "What did the speaker say about AI?").
    
    Returns:
        A dict with 'answer' (str) and 'source_documents' (list of Document objects).
    """
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    return {"answer": answer, "source_documents": docs}

rag_chain_with_sources_func = RunnableLambda(rag_chain_with_sources)

# ==================== Modern LangGraph Agent with Memory ====================
# Define state (tracks messages)
class AgentState(dict):
    messages: list

# Tool for RAG (same as before)
def qa_tool(state: AgentState):
    question = state["messages"][-1].content
    result = rag_chain_with_sources(question)
    return {
        "messages": state["messages"] + [AIMessage(content=result["answer"])]
    }

tools = [qa_tool]  # Wrap RAG as a tool

# Build the graph
def build_agent():
    graph = StateGraph(state_schema=AgentState)
    
    # Add nodes
    graph.add_node("agent", create_react_agent(chat_model, tools))
    graph.add_node("qa_tool", qa_tool)
    
    # Edges (agent decides to call tool or end)
    graph.add_conditional_edges(
        "agent",
        lambda state: "qa_tool" if "tool" in state else END,
        {"qa_tool": "agent", END: END}
    )
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add memory checkpoint
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    
    return app

# ==================== Initialize Agent ====================
def get_agent():
    return build_agent()

# ==================== Optional: Direct QA ====================
get_qa_chain = lambda: rag_chain_with_sources_func