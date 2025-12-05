# server.py
import gradio as gr
from utils import get_agent

agent_executor = get_agent()

def chat(message, history):
    if not message.strip():
        return history + [["", "Please ask a question."]]

    config = {"configurable": {"thread_id": "gradio_session"}}  # one memory per browser tab

    result = agent_executor.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )
    answer = result["messages"][-1]["content"]

    history.append([message, answer])
    return history, ""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# YouTube Transcript ChatBot")
    chatbot = gr.Chatbot(height=600)
    msg = gr.Textbox(placeholder="Ask anything about the videos...", label="Question")
    clear = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    demo.launch()

