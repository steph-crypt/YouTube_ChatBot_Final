#!/usr/bin/env python
# coding: utf-8

# ### Setup and load environments

# In[15]:


# 1. Setup
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore  # ✅ NEW correct import
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from getpass import getpass

# Load environment variables
_ = load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or getpass("Enter your Pinecone API key: ")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API key: ")


# ### Get data from pinecone after storing successfully

# In[ ]:


def reconnect_vectorstore(index_name="youtube-transcripts"):
    from langchain.embeddings import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    from dotenv import load_dotenv
    import os

    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)

    # 🔍 Print vector count directly
    stats = index.describe_index_stats()
    print(f"📦 Index contains {stats['total_vector_count']} vectors")

    vectordb = PineconeVectorStore(index, embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    docs = retriever.get_relevant_documents("What is this video about?")
    print(f"Retrieved {len(docs)} docs")
    for d in docs:
        print("-----")
        print(d.page_content)

    print("✅ Reconnected to Pinecone vectorstore and retriever is ready")
    return vectordb, retriever



# ### Set up retriever

# In[ ]:


# 4. Set up retriever
vectordb, retriever = reconnect_vectorstore()
print("✅ Retriever initialized")

# 5. Memory & Tools
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA


llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=OPENAI_API_KEY)
print("✅ Initialized LLM")

# Define RetrievalQA chain
from utils import init_qa_chain
qa_chain = init_qa_chain()

print("✅ RetrievalQA chain created")


# ### Define function for answering questions with sources

# In[26]:


def answer_with_sources(input_text: str):
    print(f"\n📝 Query: {input_text}")
    result = qa_chain({"query": input_text})
    answer = result["result"]
    sources = list(set(doc.metadata.get("source_file", "Unknown") for doc in result["source_documents"]))
    print(f"🗒️ Retrieved {len(sources)} source documents")
    return f"{answer}\n\nSources:\n" + "\n".join(sources)


# ### Setup tools and agent with Langsmith tracing

# In[27]:


from langsmith import traceable
import os
# !pip install -U langsmith openai

# Set LangSmith env vars
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "LANGCHAIN_API_KEY"
os.environ["LANGCHAIN_PROJECT"] = "youtube-rag"

from langchain.agents import initialize_agent

# Define tool
tools = [
    Tool(
        name="YouTubeTranscriptQA",
        func=answer_with_sources,
        description="Useful for answering questions about YouTube video transcripts. Input should be a fully formed question."
    )
]
print("✅ Tools defined")

# Conversation memory
memory = ConversationBufferMemory(memory_key="chat_history")
print("✅ Conversation memory initialized")

# 6. Initialize Agent with tools and memory
agent = initialize_agent(
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    memory=memory,
    max_iterations=3
)

print("✅ Agent initialized and ready")

@traceable(name="YouTube RAG Trace")
def run_agent():
    agent.run("Who was Albert Einstein?")

result = run_agent()


# ### Prompt agent

# In[28]:


prompt_templates = {
    "summary": "Please provide a concise summary of the following topic: '{}'",
    "source": "Please provide the video source for the following topic: '{}'",
    "explanation": "Explain in detail: '{}'",
    "compare": "Compare and contrast these two concepts: '{}' and '{}'",
    "timeline": "Give me a timeline of events related to: '{}'",
    "faq": "What are the most frequently asked questions about '{}', and their answers?",
    "key_points": "List the key points covered in: '{}'",
    "step_by_step": "Provide a step-by-step guide on how to: '{}'",
    "examples": "Give me examples related to '{}'",
    "pros_cons": "What are the pros and cons of '{}'",
    "common_mistakes": "What are the common mistakes people make regarding '{}', and how to avoid them?",

}

def ask_agent(agent, prompt_type, *args):
    if prompt_type not in prompt_templates:
        raise ValueError(f"Prompt type '{prompt_type}' not supported.")
    prompt = prompt_templates[prompt_type].format(*args)
    print(f"Prompt to model:\n{prompt}\n")
    response = agent.run(prompt)
    return response



# ### User query testing

# In[29]:


result = ask_agent(agent, "compare", "string theory", "pinpoint theory")
print(result)


# In[30]:


result = ask_agent(agent, "explanation", "Black holes")
print(result)


# In[31]:


result = ask_agent(agent, "source", "Black holes")
print(result)


# In[32]:


result = ask_agent(agent, "source", "string theory")
print(result)

