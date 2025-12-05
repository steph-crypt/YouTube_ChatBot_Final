# server.py
import gradio as gr
from utils import get_agent, rag_chain_with_sources 

# This keeps chat history across messages, answers naturally, and can say "I don't know"
agent_executor = get_agent()


def chat_with_agent(message: str, history):
    """
    Gradio chatbot function — keeps full conversation history automatically.
    """
    if not message.strip():
        return history + [["", "Please ask a question."]]

    # Agent returns a dict with 'output' key
    result = agent_executor.invoke({"input": message})
    answer = result["output"]

    # Append user + bot messages to history (Gradio format)
    history = history or []
    history.append([message, answer])
    return history


# ==================== Gradio Interface ====================
with gr.Blocks(title="YouTube Transcript ChatBot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # YouTube Transcript ChatBot
        Ask anything about the videos stored in your Pinecone index.
        """
    )

    # Best UI for conversation: built-in Chatbot component
    chatbot = gr.Chatbot(height=600, show_label=False)

    with gr.Row():
        msg = gr.Textbox(
            label="Your question",
            placeholder="e.g. What did Lex say about AI alignment?",
            scale=4
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    # Optional: Clear button
    clear = gr.Button("Clear conversation")

    # Connect the agent version (recommended)
    msg.submit(chat_with_agent, [msg, chatbot], chatbot], [chatbot])
    send_btn.click(chat_with_agent, [msg, chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

    # If you prefer the simple version instead, replace the above with:
    # msg.submit(lambda q: answer_question(q), msg, chatbot)
    # send_btn.click(lambda q: answer_question(q), msg, chatbot)

if __name__ == "__main__":
    demo.launch()

