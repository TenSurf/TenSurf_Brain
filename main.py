import gradio as gr
import logging
import merged
from merged import *


logging.basicConfig(filename="info.log",
                    filemode='a',
                    encoding='utf-8',
                    format='%(asctime)s %(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

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
    demo.launch(debug=True, server_port=8081, share_server_protocol="http")
