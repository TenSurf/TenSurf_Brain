from typing import Optional
import function_calling
import file_processor


def main(input_json: dict) -> str:
    file_path = input_json["file_path"]
    prompt = input_json["prompt"]
    messages = input_json["messages"]
    front_json = input_json["front_json"]
    content = ""
    file_exist = 0
    functionCalling = function_calling.FunctionCalling()
    fileProcessor = file_processor.FileProcessor()
    if file_path:
        file_exist = 1
        file_content = fileProcessor.get_content(file_path)
        if file_content:
            content += file_content + "\n"
    if prompt:
        content += prompt + "\n"
    else:
        content += fileProcessor.default_prompt + '\n'
    return functionCalling.generate_answer(messages=messages,content=content.strip(), front_json=front_json, file_exist=file_exist)
