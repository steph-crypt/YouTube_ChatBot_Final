import gradio as gr
from utils import init_qa_chain

qa_chain = init_qa_chain()

def answer_question(question: str) -> str:
    if not question.strip():
        return "Please ask a question."
    result = qa_chain.invoke({"query": question})
    return result["result"]

with gr.Blocks() as demo:
    gr.Markdown("## 🎥 YouTube Transcript ChatBot")
    question_input = gr.Textbox(
        label="Ask a question",
        placeholder="Ask something about the YouTube transcripts..."
    )
    answer_output = gr.Textbox(
        label="Answer",
        lines=6
    )
    question_input.submit(
        answer_question,
        inputs=question_input,
        outputs=answer_output
    )

demo.launch(
    share=True
)

app = demo
