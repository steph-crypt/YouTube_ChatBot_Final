#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pip', 'install -qU langchain-pinecone pinecone-notebooks')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain-text-splitters tiktoken')
get_ipython().run_line_magic('pip', 'install langchain-openai')
get_ipython().run_line_magic('pip', 'install datasets')


# ### Connect to OpenAI

# ### Retrieve Transcripts Dataframe

# ### Setup and load environments

# In[ ]:


# 1. Setup
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
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


# ### Chunk data

# In[5]:


import math
from langchain.schema import Document

# 1. Load the entire dataframe (no head())
df = pd.read_pickle("/Users/test/Desktop/ironhack_labs/YouTube_ChatBot_Final/datasets/dataframe.pkl")
print(f"✅ Loaded dataframe with {len(df)} rows")
print(df.head(2))  # preview first 2 rows

# 2. Prepare documents and sources
texts = df["text"].tolist()
sources = df["source_file"].tolist()
print(f"✅ Prepared {len(texts)} texts and sources")

# 3. Split texts into chunks with metadata
splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=200)
documents = [Document(page_content=t, metadata={"source_file": s}) for t, s in zip(texts, sources)]
print(f"✅ Created {len(documents)} Document objects")

split_docs = splitter.split_documents(documents)
print(f"✅ After splitting, got {len(split_docs)} document chunks")
print(f"Sample chunk content:\n{split_docs[0].page_content}")
print(f"Sample chunk metadata:\n{split_docs[0].metadata}")



# ### Create Pinecone index, embeddings and vectors

# In[ ]:


# 4. Initialize Pinecone client and index
index_name = "youtube-transcripts"
dimension = 1536

pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Created index: {index_name}")
else:
    print(f"✅ Index '{index_name}' already exists.")

index = pc.Index(index_name)
print(f"✅ Connected to Pinecone index: {index_name}")

# 5. Initialize OpenAI embeddings
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
print("✅ Initialized OpenAI embeddings")

# 6. Create PineconeVectorStore instance
vectordb = PineconeVectorStore(index, embedding)

# 7. Upload in smaller batches manually to avoid request size limit errors
from tqdm import tqdm
import math
batch_size = 30  # Adjust batch size if needed to avoid errors
num_batches = math.ceil(len(split_docs) / batch_size)

print(f"Total document chunks: {len(split_docs)}")
print(f"Uploading in batches of {batch_size}, total batches: {num_batches}")

for i in tqdm(range(num_batches), desc="Uploading batches"):
    batch_docs = split_docs[i * batch_size : (i + 1) * batch_size]
    vectordb.add_documents(batch_docs)

print(f"✅ Successfully stored all {len(split_docs)} document chunks in Pinecone")

