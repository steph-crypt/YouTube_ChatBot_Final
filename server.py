# server.py
import gradio as gr
from utils import get_agent

# Create agent once at startup
agent_executor = get_agent()

def chat(message, history):
    if not message.strip():
        return history, ""

    config = {"configurable": {"thread_id": "youtube_chat_session"}}

    # LangGraph expects a list of messages
    result = agent_executor.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )
    answer = result["messages"][-1]["content"]

    history.append([message, answer])
    return history, ""   # also clears the textbox

with gr.Blocks(theme=gr.themes.Soft(), title="YouTube Transcript ChatBot") as demo:
    gr.Markdown("# YouTube Transcript ChatBot")
    chatbot = gr.Chatbot(height=600)
    msg = gr.Textbox(
        placeholder="Ask anything about the videos...",
        label="Your question",
        scale=7
    )
    clear = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    demo.launch()

