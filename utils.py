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
    model="gpt-4o-mini",          # change if you want gpt-4-turbo → "gpt-4-turbo"
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ==================== Vector Store + Retriever ====================
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"               # change to "content" or whatever your field is called
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ==================== Memory ====================
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

# Fixed: proper function, not broken lambda syntax
def rag_chain_with_sources(question: str):
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    return {"answer": answer, "source_documents": docs}

# Make it callable exactly like a normal chain
rag_chain_with_sources_func = RunnableLambda(rag_chain_with_sources)

# ==================== Agent Tool ====================
qa_tool = Tool(
    name="YouTubeTranscriptQA",
    description="Useful for answering any question about YouTube video transcripts. Input must be a full question.",
    func=rag_chain_with_sources
)

# ==================== Initialize Conversational Agent ====================
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
    )
    return agent

# ==================== Optional: Direct QA (no agent) ====================
# Use this if you ever want a simple stateless chain
get_qa_chain = lambda: rag_chain_with_sources_func   # now correct syntax