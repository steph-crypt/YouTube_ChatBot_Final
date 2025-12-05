# server.py
import gradio as gr
from langchain_core.messages import HumanMessage
from utils import get_agent

# Create agent once
agent_executor = get_agent()

def chat(message, history):
    if not message.strip():
        return history, ""

    config = {"configurable": {"thread_id": "hf_space_session"}}

    result = agent_executor.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    answer = result["messages"][-1].content

    history.append([message, answer])
    return history, ""

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# YouTube Transcript ChatBot")
    gr.Markdown("Ask anything about the videos in your Pinecone index!")

    chatbot = gr.Chatbot(height=620)
    msg = gr.Textbox(
        placeholder="e.g. What did Lex say about AI safety?",
        label="Your Question"
    )
    clear = gr.Button("Clear Chat")

    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()