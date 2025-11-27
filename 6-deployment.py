import gradio as gr
import whisper
import uuid
import os
from gtts import gTTS
from dotenv import load_dotenv
import warnings
from pathlib import Path
import time
warnings.filterwarnings("ignore")

import utils, importlib
importlib.reload(utils)
from utils import init_qa_chain, get_agent


load_dotenv()

# One-time init
qa_chain = init_qa_chain()
agent = get_agent([qa_chain])
whisper_model = whisper.load_model("tiny")  # tiny = fast, base = accurate
os.makedirs("outputs", exist_ok=True)

# ──────────────────────
# Core functions
# ──────────────────────
def transcribe(audio_path):
    return whisper_model.transcribe(audio_path)["text"].strip()

def speak(text):
    file = f"outputs/{uuid.uuid4().hex}.mp3"
    gTTS(text=text, lang="en").save(file)
    return file

def answer(question):
    try:
        return agent.run(question)
    except:
        return qa_chain.run(question)

def gradio_qa(text=None, audio=None):
    if audio:
        question = transcribe(audio)
    elif text and text.strip():
        question = text.strip()
    else:
        return "Please speak or type a question…", "", None

    resp = answer(question)
    # Keep spoken answer short & clear
    short = resp[:450] + "…" if len(resp) > 450 else resp
    audio_file = speak(short)

    return question, resp, audio_file

# ──────────────────────
# EMBEDDED CSS (Gradio 5.x/6.x way – no 'css' or 'theme' in Blocks)
# ──────────────────────

embedded_css = """
<style>
    body {
        background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.85));
        color: #e0e0e0;
        min-height: 100vh;
        margin: 0;
    }

    h1 {
        font-size: 6.2rem;
        text-align: center;
        background: linear-gradient(90deg, #00ff88, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5em 0;
        text-shadow: 0 0 30px rgba(0,255,136,0.5);
    }

    footer { display: none !important; }

    /* Style for the logo/image at the top */
    .cosmic-logo {
        display: block;
        width: 180px;
        height: 180px;
        margin: 0 auto 1rem;
        border-radius: 50%;
        border: 4px solid #00ff88;
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.8);
        object-fit: cover;
    }

    .about-img {
        border-radius: 16px;
        border: 4px solid #00ffff;
        box-shadow: 0 0 40px rgba(0,255,255,0.7);
    }
</style>
"""

with gr.Blocks(title="Chat DeGrasse Tyson") as demo:
    image_path = Path("neil_planets12b_dj1_custom-e9a1db0895b141221d00733a2d5e182dc77a312e.jpg")
    neil_image_path = image_path.as_posix()

    gr.HTML(f"<style>/* refreshed at {time.time()} */ {embedded_css}</style>")

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

    # ABOUT SECTION
    with gr.Accordion("About This App", open=False):
        gr.Markdown("""
        ## Chat DeGrasse Tyson

        An AI-powered cosmic assistant that lets you **talk directly to Neil deGrasse Tyson** (well… almost).

        Ask any question about physics, space, the universe, philosophy, or life — and get thoughtful, accurate answers **spoken aloud** in a calm, cosmic voice.

        ### How it works:
        - **You speak or type** → Whisper converts speech to text  
        - **Llama-3 70B** (via Groq) generates Neil-style answers  
        - **gTTS** turns the answer into audio so you can *hear* the cosmos speak  

        Built with love for science, wonder, and exploration.

        *Made by a fan of the universe — for fans of the universe.*
        """)
        gr.Image(neil_image_path, label="Neil among the planets", height=420, elem_classes="about-img")

    # Connect everything
    send_btn.click(gradio_qa, inputs=[textbox, mic], outputs=[question_out, answer_box, audio_out])
    textbox.submit(gradio_qa, inputs=[textbox, mic], outputs=[question_out, answer_box, audio_out])

# ──────────────────────
# Launch with theme (Gradio 6.x way)
# ──────────────────────
if __name__ == "__main__":
    demo.launch(
        theme="soft",        
        share=True,        
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[image_path.parent.as_posix()]
    )