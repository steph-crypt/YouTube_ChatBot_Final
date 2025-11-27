# api.py – FastAPI backend for React
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import whisper
import uuid
import os
from gtts import gTTS
import utils
import importlib
importlib.reload(utils)
from utils import init_qa_chain, get_agent

# === Init ===
qa_chain = init_qa_chain()
agent = get_agent([qa_chain])
whisper_model = whisper.load_model("tiny")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("public", exist_ok=True)
app.mount("/static", StaticFiles(directory="public"), name="static")

def get_answer(question: str) -> str:
    try:
        return agent.run(question)
    except:
        return qa_chain.run(question)

def shorten(text: str) -> str:
    return text[:500] + "..." if len(text) > 500 else text

@app.post("/ask")
async def ask(question: str = Form(None), audio: UploadFile = File(None)):
    if audio:
        path = f"/tmp/{uuid.uuid4().hex}.wav"
        with open(path, "wb") as f:
            f.write(await audio.read())
        question = whisper_model.transcribe(path)["text"].strip()

    if not question:
        return JSONResponse({"error": "No question"})

    full = get_answer(question)
    short = shorten(full)

    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = f"public/{filename}"
    gTTS(short, lang="en").save(filepath)

    return {
        "question": question,
        "answer": short,
        "audio_url": f"http://localhost:8000/static/{filename}"
    }