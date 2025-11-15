#!/usr/bin/env python
# coding: utf-8

# # Voice input and output feature

# In[1]:


import importlib
import utils
importlib.reload(utils)
from utils import get_agent


# In[2]:


from utils import init_qa_chain, get_agent

qa_chain = init_qa_chain()
agent = get_agent(qa_chain)


# In[ ]:


#voice input
# !pip install openai-whisper pyaudio pyttsx3
# !pip install sounddevice wavio


# ### Creates user audio recording functionality

# In[3]:


import whisper
import sounddevice as sd
import numpy as np
import wavio
from gtts import gTTS
from playsound import playsound
import os
import uuid
import openai


import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


# Load Whisper model
model = whisper.load_model("base")

def record_audio(duration=5, fs=16000):
    print("🎙️ Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write("/Users/test/Desktop/ironhack_labs/YouTube_ChatBot_Final/audio/user_input.wav", recording, fs, sampwidth=2)
    print("🎙️ Recording finished")

def transcribe_audio(filepath="/Users/test/Desktop/ironhack_labs/YouTube_ChatBot_Final/audio/user_input.wav"):
    print("📝 Transcribing...")
    result = model.transcribe(filepath)
    text = result["text"].strip()
    print(f"🗣️ You said: {text}")
    return text

def speak_text(text):
    filename = f"{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

def shorten_text(text, max_ratio=0.5):
    # Prompt the agent or LLM to summarize or shorten the text
    prompt = f"Please shorten the following text to about {int(max_ratio*100)}% of its length, keeping the key information:\n\n{text}"
    shortened = agent.run(prompt)
    return shortened

def answer_with_sources(question):
    # full_answer = agent.run(question)
    prompt = f"Use the YouTubeTranscriptQA tool to answer this question: {question}"
    full_answer = agent.run(prompt)
    short_answer = shorten_text(full_answer)
    return short_answer

# One interaction
def voice_rag_interaction():
    record_audio()
    question = transcribe_audio()
    answer = answer_with_sources(question)
    print(f"🤖 Answer:\n{answer}")
    speak_text(answer)


# In[4]:


while True:
    voice_rag_interaction()
    cont = input("Ask another question? (y/n): ")
    if cont.strip().lower() != "y":
        break

