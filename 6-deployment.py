#!/usr/bin/env python
# coding: utf-8

# # Deploy chatbot to Gradio

# In[ ]:


# 📦 Imports
import gradio as gr
import whisper
import uuid
import os
from gtts import gTTS
import utils  # Import the whole module so you can reload it
import importlib
from dotenv import load_dotenv
load_dotenv()

importlib.reload(utils)

# Then you can safely import the individual functions again
from utils import init_qa_chain, get_agent


# ✅ Load QA agent
qa_chain = init_qa_chain()
agent = get_agent(qa_chain)

# ✅ Load Whisper model once
model = whisper.load_model("base")

# 🔧 Utility: Transcribe audio file using Whisper
def transcribe_audio_file(audio_file):
    print("📝 Transcribing audio...")
    result = model.transcribe(audio_file.name)
    text = result["text"].strip()
    print(f"🗣️ Transcribed text: {text}")
    return text

# 🔊 Utility: Generate TTS file using gTTS
def generate_tts(text):
    filename = f"{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

def shorten_text(text, max_ratio=0.5):
    prompt = f"Please shorten the following text to about {int(max_ratio*100)}% of its length, keeping the key information:\n\n{text}"
    shortened = agent.run(prompt)
    return shortened

# 🧠 Ask the agent
def answer_with_sources(question):
    prompt = f"Use the YouTubeTranscriptQA tool to answer this question: {question}"
    full_answer = agent.run(prompt)
    short_answer = shorten_text(full_answer)
    return short_answer

# 🧩 Gradio logic: handles text or audio
def gradio_qa(text_input, audio_input):
    if text_input and text_input.strip():
        question = text_input.strip()
        print("✅ Using text input")
    elif audio_input is not None:
        question = transcribe_audio_file(audio_input)
        print("✅ Using audio input")
    else:
        return "Please enter a question or upload audio.", "", None

    answer = answer_with_sources(question)
    audio_path = generate_tts(answer)
    return question, answer, audio_path

# 🎛️ Gradio interface
iface = gr.Interface(
    fn=gradio_qa,
    inputs=[
        gr.Textbox(label="Enter your question (text)", lines=2, placeholder="Type your question here..."),
        gr.Audio(type="filepath", label="Or upload your question as audio")
    ],
    outputs=[
        gr.Textbox(label="Processed Question"),
        gr.Textbox(label="Answer"),
        gr.Audio(label="Answer Audio", type="filepath")
    ],
    title="YouTube Transcript QA (Text & Voice Input)",
    description="Ask questions via text or audio and get answers based on YouTube transcripts, including spoken output.",
    allow_flagging="never"
)

# 🟢 Launch interface
if __name__ == "__main__":
    iface.launch(share=True)


