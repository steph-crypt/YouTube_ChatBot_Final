import gradio as gr
import whisper
import uuid
import os
from gtts import gTTS
from dotenv import load_dotenv
import warnings
import time



warnings.filterwarnings("ignore")

# Load environment variables (HF Spaces supports .env or Secrets)
load_dotenv()

# One-time init
from utils import init_qa_chain, get_agent 
qa_chain = init_qa_chain()
agent = get_agent([qa_chain])

# Use a smaller/faster Whisper model for HF (free tier has limited RAM/CPU)
whisper_model = whisper.load_model("tiny")  

os.makedirs("outputs", exist_ok=True)

# ──────────────────────
# Core functions
# ──────────────────────
def transcribe(audio_path):
    if audio_path is None:
        return ""
    return whisper_model.transcribe(audio_path)["text"].strip()

def speak(text):
    if not text.strip():
        return None
    file = f"outputs/{uuid.uuid4().hex}.mp3"
    gTTS(text=text, lang="en").save(file)
    return file

def answer(question):
    try:
        return agent.run(question)
    except Exception as e:
        print(f"Agent failed: {e}")
        return qa_chain.run(question)

def gradio_qa(text=None, audio=None):
    if audio:
        question = transcribe(audio)
    elif text and text.strip():
        question = text.strip()
    else:
        return "Please speak or type a question…", "", None

    if not question:
        return "I didn't catch that. Try again?", "", None

    resp = answer(question)
    short = resp[:450] + "…" if len(resp) > 450 else resp
    audio_file = speak(short)

    return question, resp, audio_file


# ──────────────────────
# Gradio Interface
# ──────────────────────
with gr.Blocks(css="style.css", title="Chat DeGrasse Tyson") as demo:
    gr.HTML(f"""
        <div style="text-align:center; padding:2rem;">
            <h1>Chat DeGrasse Tyson</h1>
            <p style="font-size:1.5rem; color:#00ff88;">
                Ask me anything about the universe — with your voice
            </p>
        </div>
    """)

    with gr.Row():
        textbox = gr.Textbox(
            placeholder="Type here or click the mic…",
            lines=3,
            label="Your Question"
        )
        mic = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Click & Speak",
            waveform_options=False
        )

    send_btn = gr.Button("Ask the Universe", variant="primary", size="lg")

    with gr.Accordion("Answer from the Cosmos", open=True):
        question_out = gr.Textbox(label="You asked", interactive=False)
        answer_box = gr.Textbox(label="Answer", lines=11)
        audio_out = gr.Audio(label="Spoken Answer", autoplay=True)

    gr.Examples(
        examples=[
            ["What is dark matter?"],
            ["Explain quantum entanglement like I'm 10"],
            ["Are we alone in the universe?"],
            ["Why is the sky blue?"]
        ],
        inputs=textbox
    )

    # About Section
    with gr.Accordion("About This App", open=False):
        gr.Markdown("""
        ## Chat DeGrasse Tyson

        An AI-powered cosmic assistant that lets you **talk directly to Neil deGrasse Tyson** (well… almost).

        Ask any question about physics, space, the universe, philosophy, or life — and get thoughtful, accurate answers **spoken aloud** in a calm, cosmic voice.

        ### How it works:
        - **You speak or type** → Whisper converts speech to text  
        - **Llama-3 70B** (via Groq) or fallback to RetrievalQA generates answers  
        - **gTTS** turns the answer into audio so you can *hear* the cosmos speak  

        Built with love for science, wonder, and exploration.

        *Made by a fan of the universe — for fans of the universe.*
        """)

        # Safe image loading
        image_path = "neil_planets12b_dj1_custom-e9a1db0895b141221d00733a2d5e182dc77a312e.jpg"
        if os.path.exists(image_path):
            gr.Image(image_path, label="Neil among the planets", height=420, elem_classes="about-img")
        else:
            gr.Markdown("_Image not available in this deployment_")

    # Events
    send_btn.click(gradio_qa, inputs=[textbox, mic], outputs=[question_out, answer_box, audio_out])
    textbox.submit(gradio_qa, inputs=[textbox, mic], outputs=[question_out, answer_box, audio_out])
    mic.change(gradio_qa, inputs=[textbox, mic], outputs=[question_out, answer_box, audio_out])  

