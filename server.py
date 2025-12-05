# server.py
import gradio as gr
from utils import get_agent, rag_chain_with_sources

# Initialize modern agent
agent_executor = get_agent()

def chat_with_agent(message: str, history):
    if not message.strip():
        return history + [["", "Please ask a question."]]

    # LangGraph input: config for memory (session_id keeps history)
    config = {"configurable": {"thread_id": "conversation_1"}} 
    
    # Invoke with messages
    result = agent_executor.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    
    # Extract last AI message
    answer = result["messages"][-1].content

    history = history or []
    history.append([message, answer])
    return history, ""  

# Gradio Interface (same as before)
with gr.Blocks(title="YouTube Transcript ChatBot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# YouTube Transcript ChatBot")
    chatbot = gr.Chatbot(height=600, show_label=False)
    with gr.Row():
        msg = gr.Textbox(label="Your question", placeholder="e.g. What did Lex say about AI?", scale=4)
        send_btn = gr.Button("Send", variant="primary", scale=1)
    clear = gr.Button("Clear conversation")
    
    msg.submit(chat_with_agent, [msg, chatbot], [chatbot, msg])
    send_btn.click(chat_with_agent, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: (None, ""), None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()

