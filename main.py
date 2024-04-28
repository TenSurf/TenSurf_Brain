import gradio as gr
from merged import *
from utils import messages
from config import *


input_audio = gr.Audio(sources="microphone", show_download_button=True, type="filepath")


def random_response(message, history, audio):
    text_message = message["text"]
    file_message = message["files"]
    llm = FileProcessor()
    result = llm.get_user_input(file_path=None, prompt=text_message, messages=messages)
    return result


demo = gr.ChatInterface(
    fn=random_response, title="Function Calling", multimodal=True,
    additional_inputs=input_audio,
)


if __name__ == "__main__":
    demo.launch(debug=True, server_port=8081, share_server_protocol="http", root_path="/api/v1")
