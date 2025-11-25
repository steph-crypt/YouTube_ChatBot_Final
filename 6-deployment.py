import gradio as gr
import whisper
import uuid
import os
from gtts import gTTS
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress LangChain warnings

# Reload and import utils
import utils
import importlib
importlib.reload(utils)
from utils import init_qa_chain, get_agent

# Load environment variables
load_dotenv()

# Initialize your QA system ONCE at startup
print("🚀 Initializing QA chain...")
qa_chain = init_qa_chain()
print("✅ QA chain ready!")

# Try agent, fallback to direct chain if agent fails
try:
    agent = get_agent([qa_chain])  # Pass as list for tool wrapping
    print("✅ Agent ready!")
except Exception as e:
    print(f"⚠️ Agent init failed ({e}), using direct QA chain.")
    agent = None  # Fallback

# Load Whisper model ONCE (use "tiny" for speed)
print("🎤 Loading Whisper model...")
whisper_model = whisper.load_model("tiny")
print("✅ Whisper ready!")

# === Utility Functions ===
def transcribe_audio_file(audio_path):
    print("🎙️ Transcribing audio...")
    result = whisper_model.transcribe(audio_path)
    text = result["text"].strip()
    print(f"🗣️ You said: {text}")
    return text

def generate_tts(text):
    filename = f"answer_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("outputs", filename)
    os.makedirs("outputs", exist_ok=True)
    tts = gTTS(text=text, lang="en")
    tts.save(filepath)
    return filepath

def get_answer(question):
    """Get answer using agent or direct chain."""
    if agent:
        try:
            return agent.run(question)
        except:
            print("⚠️ Agent failed, falling back to QA chain.")
    return qa_chain.run(question)

def shorten_answer(text):
    # Simple truncation for voice (or use LLM if agent works)
    if len(text) > 200:
        return f"{text[:197]}..."
    return text

def answer_with_sources(question):
    print(f"❓ Question: {question}")
    full_answer = get_answer(question)
    short_answer = shorten_answer(full_answer)
    return short_answer

# === Main Gradio Function ===
def gradio_qa(text_input=None, audio_input=None):
    # Determine question source
    if audio_input is not None:
        question = transcribe_audio_file(audio_input)
    elif text_input and text_input.strip():
        question = text_input.strip()
    else:
        return "Please speak or type a question.", "", None

    # Get answer
    answer = answer_with_sources(question)
    
    # Generate spoken response
    audio_file = generate_tts(answer)

    return question, answer, audio_file

# === Gradio Interface (Gradio 5.x Compatible) ===
demo = gr.Interface(
    fn=gradio_qa,
    inputs=[
        gr.Textbox(
            label="Type your question",
            placeholder="e.g. What is a black hole?",
            lines=3
        ),
        gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Or click to record your voice"
        )
    ],
    outputs=[
        gr.Textbox(label="Your Question"),
        gr.Textbox(label="Answer", lines=10),
        gr.Audio(label="Spoken Answer", autoplay=True)
    ],
    title="YouTube Science Tutor",
    description="Ask anything from your YouTube transcripts — with voice!",
    theme=gr.themes.Soft(),
    allow_flagging="never",
    examples=[
        ["What is quantum entanglement?"],
        ["Explain string theory simply"],
        ["Who is Neil deGrasse Tyson?"]
    ]
)

# === Launch ===
if __name__ == "__main__":
    print("Launching your AI voice tutor...")
    demo.launch(
        share=True,           # Gives you a public link
        server_name="0.0.0.0",
        server_port=7860
    )