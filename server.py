import gradio as gr
from utils import init_qa_chain

# Initialize QA chain
qa_chain = init_qa_chain()

def answer_question(question: str):
    """Return answer from QA chain."""
    if not question.strip():
        return "Please ask a question."
    result = qa_chain.run(question)
    return result

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## YouTube Transcript ChatBot")
    question_input = gr.Textbox(label="Ask a question")
    answer_output = gr.Textbox(label="Answer")
    question_input.submit(answer_question, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    demo.launch()

