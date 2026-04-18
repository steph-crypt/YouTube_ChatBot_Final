import os
import tempfile
import uuid
import warnings
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import whisper
from dotenv import load_dotenv
from gtts import gTTS

warnings.filterwarnings("ignore")

# Load .env from project root regardless of cwd
load_dotenv(Path(__file__).resolve().parent / ".env")

# One-time init
from utils import get_agent, init_qa_chain

qa_chain = init_qa_chain()
agent = get_agent([qa_chain])

# Whisper — local STT (tiny = fast; use "base" for better accuracy if CPU allows)
whisper_model = whisper.load_model("tiny")

os.makedirs("outputs", exist_ok=True)


def transcribe(audio_in):
    """
    Gradio may pass a temp file path (type='filepath') or (sample_rate, numpy_array) if type is numpy.
    Whisper's file path path uses ffmpeg resampling; for raw arrays we write a temp WAV first.
    """
    if audio_in is None:
        return ""

    if isinstance(audio_in, str) and os.path.isfile(audio_in):
        return whisper_model.transcribe(audio_in)["text"].strip()

    if isinstance(audio_in, (tuple, list)) and len(audio_in) == 2:
        sr, data = audio_in
        data = np.asarray(data)
        if data.size == 0:
            return ""
        if data.ndim > 1:
            data = data.mean(axis=1)
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp, data, int(sr))
            return whisper_model.transcribe(tmp)["text"].strip()
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    return ""


def speak(text: str):
    if not text or not str(text).strip():
        return None
    out = Path("outputs") / f"{uuid.uuid4().hex}.mp3"
    gTTS(text=str(text)[:5000], lang="en").save(str(out))
    return str(out.resolve())


def answer(question: str) -> str:
    try:
        return agent.run(question)
    except Exception as e:
        print(f"Agent failed: {e}")
        out = qa_chain.invoke({"query": question})
        return out.get("result", str(out))


def _clear_inputs():
    """Reset text + mic so the previous question’s audio file path isn’t reused."""
    return gr.update(value=""), gr.update(value=None)


def gradio_qa(text=None, audio=None):
    """
    Prefer typed text when present (avoids Gradio keeping the last recording filepath).
    Otherwise use transcribed audio. Clear inputs after each run so the next question is fresh.
    """
    typed = (str(text).strip() if text else "")
    has_audio = audio is not None and (
        (isinstance(audio, str) and audio.strip() != "")
        or (isinstance(audio, tuple) and len(audio) == 2 and np.asarray(audio[1]).size > 0)
    )

    if typed:
        question = typed
    elif has_audio:
        question = transcribe(audio)
    else:
        q, a, au = "Please speak or type a question…", "", None
        c1, c2 = _clear_inputs()
        return q, a, au, c1, c2

    if not question:
        q, a, au = "I didn't catch that. Try again?", "", None
        c1, c2 = _clear_inputs()
        return q, a, au, c1, c2

    resp = answer(question)
    short = resp[:450] + "…" if len(resp) > 450 else resp
    audio_file = speak(short)
    c1, c2 = _clear_inputs()
    return question, resp, audio_file, c1, c2


# ──────────────────────
# Gradio Interface (light theme)
# ──────────────────────
_light_theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(
    theme=_light_theme,
    css="style.css",
    title="Chat DeGrasse Tyson",
) as demo:
    gr.HTML("""
        <div class="hero">
            <h1 class="hero-title">Chat DeGrasse Tyson</h1>
            <p class="hero-tagline">Ask me anything about the universe — with your voice</p>
        </div>
    """)

    with gr.Row():
        textbox = gr.Textbox(
            placeholder="Type here, or use the microphone…",
            lines=3,
            label="Your Question",
        )
        mic = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Record or upload audio",
        )

    send_btn = gr.Button("Ask the Universe", variant="primary", size="lg")

    with gr.Accordion("Answer from the Cosmos", open=True):
        question_out = gr.Textbox(label="You asked", interactive=False)
        answer_box = gr.Textbox(label="Answer", lines=11)
        audio_out = gr.Audio(
            label="Spoken answer (short clip)",
            autoplay=False,
            type="filepath",
        )

    gr.Markdown(
        '<p class="tip-line">Tip: allow microphone access in your browser. After recording, click '
        "<strong>Ask the Universe</strong> (or stop recording first if your browser shows a Stop control). "
        "Use the speaker control to play the spoken summary.</p>"
    )

    gr.Examples(
        examples=[
            ["What is dark matter?"],
            ["Explain quantum entanglement like I'm 10"],
            ["Are we alone in the universe?"],
            ["Why is the sky blue?"],
        ],
        inputs=textbox,
    )

    with gr.Accordion("About This App", open=False):
        gr.Markdown("""
        ## Chat DeGrasse Tyson

        An AI-powered cosmic assistant that lets you **talk directly to Neil deGrasse Tyson** (well… almost).

        Ask any question about physics, space, the universe, philosophy, or life — and get thoughtful, accurate answers **spoken aloud** in a calm, cosmic voice.

        ### How it works:
        - **You speak or type** → Whisper converts speech to text  
        - **Claude** (Anthropic) answers using retrieved transcripts; **local Hugging Face** embeddings + Pinecone retrieve passages  
        - **gTTS** turns a short summary into audio so you can hear the answer  

        Built with love for science, wonder, and exploration.

        *Made by a fan of the universe — for fans of the universe.*
        """)

        image_path = "neil_planets12b_dj1_custom-e9a1db0895b141221d00733a2d5e182dc77a312e.jpg"
        if os.path.exists(image_path):
            gr.Image(image_path, label="Neil among the planets", height=420, elem_classes="about-img")
        else:
            gr.Markdown("_Image not available in this deployment_")

    # Do not use mic.change — it fires during recording and breaks STT.
    _qa_outputs = [question_out, answer_box, audio_out, textbox, mic]
    send_btn.click(gradio_qa, inputs=[textbox, mic], outputs=_qa_outputs)
    textbox.submit(gradio_qa, inputs=[textbox, mic], outputs=_qa_outputs)
    if hasattr(mic, "stop_recording"):
        mic.stop_recording(gradio_qa, inputs=[textbox, mic], outputs=_qa_outputs)


if __name__ == "__main__":
    demo.launch()
