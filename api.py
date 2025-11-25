from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import uuid
from gtts import gTTS
import os
from dotenv import load_dotenv
import utils
import importlib

importlib.reload(utils)
from utils import init_qa_chain, get_agent

load_dotenv()

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup
qa_chain = init_qa_chain()
agent = get_agent(qa_chain)
whisper_model = whisper.load_model("tiny")  # or "base" / "small"

class TextQuery(BaseModel):
    question: str

def shorten_answer(text: str) -> str:
    prompt = f"Summarize this in 2-3 sentences for spoken response:\n\n{text}"
    return agent.run(prompt)

def generate_speech(text: str) -> str:
    filename = f"output_{uuid.uuid4().hex}.mp3"
    path = f"./public/{filename}"
    os.makedirs("./public", exist_ok=True)
    tts = gTTS(text=text, lang="en")
    tts.save(path)
    return f"http://localhost:8000/static/{filename}"

@app.post("/ask")
async def ask_question(
    question: str = Form(None),
    audio: UploadFile = File(None)
):
    if audio:
        # Save uploaded audio
        audio_path = f"/tmp/{uuid.uuid4().hex}.wav"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        # Transcribe
        result = whisper_model.transcribe(audio_path)
        question = result["text"].strip()

    if not question:
        return {"error": "No question provided"}

    # Get answer from your agent
    full_answer = agent.run(f"Use the YouTubeTranscriptQA tool: {question}")
    short_answer = shorten_answer(full_answer)
    audio_url = generate_speech(short_answer)

    return {
        "question": question,
        "answer": short_answer,
        "full_answer": full_answer,
        "audio_url": audio_url
    }

# Serve audio files
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="public"), name="static")