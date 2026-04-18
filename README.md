---
title: Chat DegrassTyson
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# 🎥 YouTube Transcript RAG Agent

This project implements a voice-interactive RAG (Retrieval-Augmented Generation) system that enables users to query YouTube video transcripts from the Niel Degrass Tyson Startalk channel using natural language. It leverages LangChain, Pinecone, AnthropicAI, and Whisper to retrieve, embed, and answer questions based on real video content.

---

## 📁 Project Structure

The project is organized into six main Jupyter notebooks:

| Notebook | Title                      | Description |
|----------|---------------------------|-------------|
| `1_data_retrieval.ipynb` | Data Retrieval | Downloads transcripts from a YouTube playlist, cleans them, and stores them in a structured format. |
| `2_embeddings.ipynb`     | Embedding Pipeline | Splits the transcript data, generates AnthropicAI embeddings, and uploads to Pinecone. |
| `3_agent_retrieval.ipynb`| Agent Setup & Retrieval | Initializes a LangChain agent with memory and tools for intelligent document retrieval and question-answering. |
| `4_evaluation.ipynb`     | Evaluation | Evaluates QA responses using ROUGE metrics against reference answers. |
| `5_voice_implementation.ipynb` | Voice Interaction | Integrates Whisper ASR and text-to-speech for a voice-based Q&A interface. |
| `6_deployment.ipynb`     | Deployment | Deploys the voice interface as a web app using Gradio. |

---

## 🔍 Notebook Overviews

### 📘 1. `1_data_retrieval.ipynb` — *Data Retrieval*
- Downloads English subtitles from a YouTube playlist using `yt_dlp`.
- Converts `.vtt` subtitle files into `.json` and `.txt`.
- Loads all `.txt` files into a Pandas DataFrame.
- Removes consecutive duplicate lines to clean up redundant auto-transcriptions.
- Saves the final cleaned transcripts to disk for later use.

### 📘 2. `2_embeddings.ipynb` — *Embedding Pipeline*
- Loads the cleaned transcript dataset.
- Converts each line of text into `Document` objects with metadata.
- Splits text into overlapping chunks (400 chars, 200 overlap).
- Generates embeddings using `AnthropicAIEmbeddings`.
- Stores embeddings in Pinecone (with optional batch uploading for robustness).

### 📘 3. `3_agent_retrieval.ipynb` — *Agent Setup & Retrieval*
- Reconnects to the Pinecone vector store.
- Wraps the vector database with a retriever object.
- Initializes a GPT-4-powered `ConversationalReActAgent` using LangChain.
- Incorporates memory and tools, including a custom `YouTubeTranscriptQA` tool.
- Defines prompt templates for structured queries (summary, explanation, comparison, etc.).

### 📘 4. `4_evaluation.ipynb` — *Evaluation*
- Imports the `RetrievalQA` chain.
- Uses `evaluate` and `nltk` to compute ROUGE scores between model answers and human-written references.
- Useful for validating the quality of generated answers against ground truth.

### 📘 5. `5_voice_implementation.ipynb` — *Voice Interaction*
- Records audio using `sounddevice` and saves it locally.
- Uses AnthropicAI's Whisper to transcribe the audio to text.
- Answers questions using the agent and shortens the responses using summarization prompts.
- Converts answers to speech with `gTTS` and plays them back.
- Enables fully voice-based RAG interaction loop.

### 📘 6. `6_deployment.ipynb` — *Deployment*
- Wraps the voice-based RAG system in a Gradio interface.
- Uploads, transcribes, and answers voice-based questions in a user-friendly web app.
- Uses the same pipeline as notebook 5, but exposes it via web UI.

---

## 🧠 Tech Stack

- **LangChain** — Agent-based orchestration and retrieval.
- **AnthropicAI** — Language model for QA and summarization.
- **Pinecone** — Vector database for storing and retrieving embeddings.
- **Whisper** — Speech-to-text transcription.
- **gTTS** — Text-to-speech generation.
- **Gradio** — Web-based deployment for voice interaction.
- **evaluate & ROUGE** — Evaluation metrics for response quality.

---

## 🚀 Getting Started

1. **Set environment variables**:
   - `ANTHROPIC_API_KEY`
   - `PINECONE_API_KEY`
   - (Optional) `LANGCHAIN_API_KEY` for LangSmith tracing

2. **Install required libraries**:
   ```bash
   pip install -r requirements.txt


3. **Run notebooks in order**:

1_data_retrieval.ipynb

2_embeddings.ipynb

3_agent_retrieval.ipynb

(Optional) 4_evaluation.ipynb

5_voice_implementation.ipynb

(Or deploy with) 6_deployment.ipynb

###Example Use Cases
Ask detailed questions about science videos.

Compare concepts discussed in YouTube playlists.

Voice-based educational assistant.

Evaluate knowledge from specific videos or lectures.

### To Do / Improvements
Add multi-language support for subtitles.

Integrate fallback for missing transcripts.

Enhance summarization with map-reduce chains.

Enable fine-tuned agent responses per channel or topic.

Store user chat history in database for personalization.

📄 License
MIT License. Use responsibly. Attribution appreciated.

👨‍💻 Author
Developed by steph-crypt with ❤️ using LangChain, AnthropicAI, and Pinecone.

