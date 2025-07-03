# YouTube_ChatBot_Final
## Project Overview
This project builds a Retrieval-Augmented Generation (RAG) system using YouTube transcripts from Neil deGrasse Tyson's StarTalk podcast. It involves two main stages:

## Transcript Processing and Cleaning:
Raw .txt transcript files are loaded, cleaned to remove immediate repeated lines, and structured into a Pandas DataFrame. Cleaned transcripts are saved for reproducibility and easier downstream use.

## Vector Embedding and Querying:
The cleaned transcripts are chunked, embedded using OpenAI embeddings, and uploaded in batches to a Pinecone vector database. A GPT-4-powered conversational agent is then set up to answer questions based on the indexed transcripts. The agent supports text-based queries as well as voice interaction via Whisper ASR and TTS.

### Notebook 1: Transcript Processing and Upload
#### Overview
This notebook prepares raw YouTube transcript data for retrieval-based modeling by:

Loading raw transcript .txt files from a directory.

Cleaning transcripts by removing consecutive repeated lines to avoid redundant information.

Saving cleaned transcripts back to disk.

Loading cleaned transcripts into a structured DataFrame.

(Optional) Splitting documents and uploading them to Pinecone in manageable batches.

#### Features
Read and parse multiple raw transcript files.

Clean repeated consecutive lines while preserving meaningful rephrasings.

Save cleaned files into a dedicated directory.

Convert cleaned text into a Pandas DataFrame with metadata.

Batch upload documents into Pinecone vector database with progress tracking.

Setup and Dependencies
Python 3.x

Packages: tqdm, pandas, os, math, pinecone-client, langchain

Install with:

bash
Copy
Edit
pip install tqdm pandas pinecone-client langchain
Usage
python
Copy
Edit
##### Load raw transcripts
df_raw = load_txt_transcripts_to_df(transcript_dir)

##### Clean repeated lines and save cleaned transcripts
clean_repeated_lines_df_and_save(df_raw, output_clean_dir)

##### Load cleaned transcripts into DataFrame
df_clean = load_txt_files_to_dataframe(output_clean_dir)
print(df_clean.head())

##### Save for later use
df_clean.to_pickle("/path/to/save/dataframe.pkl")

##### (Optional) Batch upload to Pinecone
batch_size = 10
num_batches = math.ceil(len(split_docs) / batch_size)

for i in tqdm(range(num_batches), desc="Uploading batches"):
    batch_docs = split_docs[i * batch_size : (i + 1) * batch_size]
    vectordb.add_documents(batch_docs)
Directory Structure Example
bash
Copy
Edit
datasets/
├── transcripts/           # Raw transcript .txt files
├── cleaned_transcripts/   # Cleaned transcripts saved here
└── dataframe.pkl          # Pickled DataFrame of cleaned transcripts
Notebook 2: Embedding, Indexing, and Conversational Agent
Overview
This notebook takes the cleaned transcripts DataFrame, prepares document chunks with metadata, and creates vector embeddings with OpenAI. It builds and manages a Pinecone index to enable fast similarity search. On top of this, it sets up a LangChain conversational agent powered by GPT-4 that can answer questions using the indexed transcript data.

Additionally, it supports voice-based interaction by recording audio questions, transcribing with Whisper, and replying using text-to-speech.

Features
Load cleaned transcripts DataFrame.

Chunk texts into overlapping pieces with source metadata.

Initialize Pinecone vector index (create if missing).

Embed documents with OpenAI and batch upload to Pinecone.

Initialize Retriever and GPT-4 RetrievalQA chain.

Define a conversational agent with memory and custom QA tools.

Support multi-format prompt templates (summary, comparison, timeline, etc.).

Evaluate answers with BLEU score against references.

Voice interaction loop using Whisper for ASR and pyttsx3 for TTS.

Setup and Dependencies
Python 3.x

Packages:
pandas, langchain, pinecone-client, dotenv, tqdm, whisper, pyttsx3, sounddevice, wavio, nltk

Install with:

bash
Copy
Edit
pip install pandas langchain pinecone-client python-dotenv tqdm whisper pyttsx3 sounddevice wavio nltk
Requires API keys for Pinecone, OpenAI, and LangChain (LangSmith tracing).

Usage
python
Copy
Edit
# Load cleaned transcripts DataFrame
df = pd.read_pickle("/path/to/dataframe.pkl")

# Create LangChain Document objects with metadata
documents = [Document(page_content=text, metadata={"source_file": src}) for text, src in zip(df["text"], df["source_file"])]

# Chunk documents
split_docs = splitter.split_documents(documents)

# Initialize Pinecone index and upload document embeddings in batches
# Initialize Retriever and GPT-4 RetrievalQA chain

# Define conversational agent with tools and memory

# Run queries:
response = answer_with_sources("Who was Albert Einstein?")
print(response)

# Voice interaction loop
voice_rag_interaction()
Summary
Together, these notebooks form a complete pipeline to:

Process and clean raw YouTube transcripts.

Convert transcripts to structured, chunked documents.

Embed and index the data for fast semantic search.

Enable GPT-4 based conversational QA on podcast transcript content.

Support both text and voice input/output for natural user interaction.

This system can be extended or adapted for other video or audio transcript data sources with minimal changes.

