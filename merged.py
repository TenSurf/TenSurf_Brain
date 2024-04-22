# -*- coding: utf-8 -*-
"""Final_FileProcessing_GPT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1o_VWgXzFbICtCyLz76eeH1LYy6A6tZWF
"""

from openai import OpenAI, AzureOpenAI
import PyPDF2
import os
from typing import Optional
import requests
from typing import Optional
import base64
from openpyxl import load_workbook
import requests
import json
import csv
from pptx import Presentation
import docx
from dateutil.relativedelta import relativedelta

from chat.gpt.functions_json import functions
from chat.gpt.functions_python import *
from chat.gpt.utils import date_validation, monthdelta
from io import BufferedReader, StringIO


class FileProcessor:
    def __init__(self):
        pass

    def chat_with_ai(self, messages: list, content: str, api_name: str):
        # try:
            if not content:
                return ""
            
            if api_name == 'openai':
              api_key = "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"
              client = OpenAI(api_key=api_key)
              GPT_MODEL_3 = "gpt-3.5-turbo-1106"
            elif (not api_name) or (api_name == 'azureopenai'):
              api_type = "azure"
              api_endpoint = 'https://tensurf.openai.azure.com/'
              api_version = '2023-10-01-preview'
              api_key = '74b3de375b964f73a6b7668fe459e26f'
              client = AzureOpenAI(
                api_key= api_key,
                api_version= api_version,
                azure_endpoint= api_endpoint
                )
              GPT_MODEL_3 = "gpt_35"
            
            def get_response(messages, functions, model, function_call):
                response = client.chat.completions.create(
                    model=model, messages=messages, functions=functions, function_call=function_call)
                return response
            
            def get_result(messages, chat_response):
                assistant_message = chat_response.choices[0].message
                messages.append(assistant_message)
                
                if chat_response.choices[0].message.function_call == None:
                    results = f"{chat_response.choices[0].message.content}"
                
                else:
                    function_name = chat_response.choices[0].message.function_call.name
                    function_arguments = json.loads(chat_response.choices[0].message.function_call.arguments)
                    FC = FunctionCalls()
                    print(f"\n{chat_response.choices[0].message}\n")
                    now = datetime.now()
                    
                    if function_name == "detect_trend":
                        # correcting function_arguments
                        if "lookback" not in function_arguments:
                            if "symbol" not in function_arguments:
                                function_arguments["symbol"] = "NQ"
                            if "start_datetime" not in function_arguments:
                                function_arguments["start_datetime"] = f"{now - timedelta(days=10)}"
                            if "end_datetime" not in function_arguments:
                                function_arguments["end_datetime"] = f"{now}"
                        else:
                            if (("lookback" in function_arguments) and ("start_datetime" in function_arguments)) or \
                                (("lookback" in function_arguments) and ("end_datetime" in function_arguments)):
                                raise ValueError("Both lookback and datetimes could not be valued")
                            else:
                                function_arguments["end_datetime"] = f"{now}"
                                k = int(function_arguments["lookback"].split(" ")[0])
                                if function_arguments["lookback"].split(" ")[-1] == "seconds" or function_arguments["lookback"].split(" ")[-1] == "second":
                                    function_arguments["start_datetime"] = f"{now - timedelta(seconds=k)}"
                                elif function_arguments["lookback"].split(" ")[-1] == "minutes" or function_arguments["lookback"].split(" ")[-1] == "minute":
                                    function_arguments["start_datetime"] = f"{now - timedelta(minutes=k)}"
                                elif function_arguments["lookback"].split(" ")[-1] == "hours" or function_arguments["lookback"].split(" ")[-1] == "hour":
                                    function_arguments["start_datetime"] = f"{now - timedelta(hours=k)}"
                                elif function_arguments["lookback"].split(" ")[-1] == "days" or function_arguments["lookback"].split(" ")[-1] == "day":
                                    function_arguments["start_datetime"] = f"{now - timedelta(days=k)}"
                                elif function_arguments["lookback"].split(" ")[-1] == "weeks" or function_arguments["lookback"].split(" ")[-1] == "week":
                                    function_arguments["start_datetime"] = f"{now - timedelta(weeks=k)}"
                                elif function_arguments["lookback"].split(" ")[-1] == "months" or function_arguments["lookback"].split(" ")[-1] == "month":
                                    function_arguments["start_datetime"] = f"{monthdelta(now, -k)}"
                                elif function_arguments["lookback"].split(" ")[-1] == "years" or function_arguments["lookback"].split(" ")[-1] == "year":
                                    function_arguments["start_datetime"] = f"{now - relativedelta(years=k)}"
                                else:
                                    raise ValueError("???")

                        # if the date formats were not valid
                        if not (date_validation(function_arguments["start_datetime"]) and date_validation(function_arguments["end_datetime"])):
                            results = "Please enter dates in the following foramat: YYYY-MM-DD or specify a period of time whether for the past seconds or minutes or hours or days or weeks or years."
                        
                        trend = FC.detect_trend(parameters=function_arguments)
                        messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {trend}. At any situations, never return the number which is the output of the detect_trend function. Instead, use its correcsponding explanation which is in the detect_trend function's description. Make sure to mention the start_datetime and end_datetime. If the user provide neither specified both start_datetime and end_datetime nor lookback parameters, politely tell them that they should. Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Now generate a proper response."})
                        chat_response = get_response(
                            messages, functions, GPT_MODEL_3, "auto"
                        )
                        assistant_message = chat_response.choices[0].message
                        messages.append(assistant_message)
                        results = chat_response.choices[0].message.content
                    
                    elif function_name == "calculate_sr":
                        # correcting function_arguments
                        if "symbol" not in function_arguments:
                            function_arguments["symbol"] = "ES"
                        if "timeframe" not in function_arguments:
                            function_arguments["timeframe"] = "1h"
                        if "lookback_days" not in function_arguments:
                            function_arguments["lookback_days"] = "10 days"

                        sr_value, sr_start_date, sr_end_date, sr_importance = FC.calculate_sr(parameters=function_arguments)
                        messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {sr_value} for levels_prices, {sr_start_date} for levels_start_timestamps, {sr_end_date} for levels_end_timestamps and {sr_importance} for levels_scores. Now generate a proper response"})
                        chat_response = get_response(
                            messages, functions, GPT_MODEL_3, "auto"
                        )
                        results = chat_response.choices[0].message.content
                    
                    else:
                        raise ValueError(f"{chat_response.choices[0].message}")
                        # results = f"{chat_response.choices[0].message}"
                
                return results
            
            messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
            messages.append({"role": "user", "content": content})
            response = get_response(messages, functions, GPT_MODEL_3, "auto")
            res = get_result(messages, response)
            return res
        
        # except Exception as e:
        #     print(f"An error occurred while chatting with AI, please try again with: {e}")

    def image_process(self, file) -> str:
        try:
            print("Image file detected. VisionGPT processing...")
            # OpenAI API Key
            api_key = os.environ.get("OPENAI_API_KEY")
            # Perform VisionGPT processing here
            def encode_image(file):
                return base64.b64encode(file.read()).decode('utf-8')

            base64_image = encode_image(file)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What’s in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for any HTTP error status
            data_image = response.json()
            file_contents = data_image['choices'][0]['message']['content']
            print("Image processing complete.")
            return file_contents
        except Exception as e:
            print(f"An error occurred while processing the image: {e}")
            return ""

    def pdf_process(self, file) -> str:
        try:
            print("PDF file detected. Extracting text...")
            pdf = PyPDF2.PdfReader(file)
            file_contents = ''
            for page in range(len(pdf.pages)):
                file_contents += pdf.pages[page].extract_text()
            print("Extraction complete.")
            return file_contents
        except Exception as e:
            print(f"An error occurred while processing the PDF: {e}")
            return ""

    def text_process(self, file) -> str:
        try:
            print("Text file detected. Extracting text...")
            file_contents = file.read().decode('utf-8')
            print("Extraction complete.")
            return file_contents
        except Exception as e:
            print(f"An error occurred while processing the text file: {e}")
            return ""

    def word_process(self, file) -> str:
        try:
            print("Word document detected. Extracting text...")
            doc = docx.Document(file)
            file_contents = '\n'.join([para.text for para in doc.paragraphs])
            print("Extraction complete.")
            return file_contents
        except Exception as e:
            print(f"An error occurred while processing the Word document: {e}")
            return ""

    def powerpoint_process(self, file) -> str:
        try:
            print("Powerpoint document detected. Extracting text...")
            # Load the PowerPoint presentation
            prs = Presentation(file)
            # Initialize an empty list to store the extracted lines of text
            extracted_lines = []
            # Iterate through each slide in the presentation
            for slide in prs.slides:
                # Iterate through each shape in the slide
                for shape in slide.shapes:
                    # Check if the shape has text
                    if hasattr(shape, "text"):
                        # Strip leading and trailing spaces from the text
                        text = shape.text.strip()
                        # Append non-empty lines to the list
                        if text:
                            extracted_lines.append(text)

            # Join the extracted lines into a single string with newlines
            file_contents = '\n'.join(extracted_lines)
            print("Extraction complete.")
            # Return the extracted text
            return file_contents

        except Exception as e:
            # Print an error message if an exception occurs
            print(f"An error occurred while processing the PowerPoint file: {e}")
            return ""

    def speech_process(self, speech) -> str:
        try:
            print("Speech file detected. Processing speech...")
            client = OpenAI()
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file= BufferedReader(speech))
            speech_contents = transcription.text
            print("Speech processing complete.")
            return speech_contents
        except Exception as e:
            print(f"An error occurred while processing the speech file: {e}")
            return ""

    def excel_process(self, file) -> str:
        try:
            print("Excel file detected. Extracting text...")
            wb = load_workbook(file)
            file_contents = '\n'.join([str(cell.value) for sheet in wb.sheetnames for row in wb[sheet].iter_rows() for cell in row if cell.value is not None])
            print("Extraction complete.")
            return file_contents
        except Exception as e:
            print(f"An error occurred while processing the Excel file: {e}")
            return ""

    def csv_process(self, file) -> str:
        try:
            print("CSV file detected. Extracting text...")
            csv_reader = csv.reader(StringIO(file.read().decode('utf-8')))
            file_contents = '\n'.join(','.join(row) for row in csv_reader)
            print("Extraction complete.")
            return file_contents
        except Exception as e:
            print(f"An error occurred while processing the CSV file: {e}")
            return ""

    def process_file(self, file) -> str:
        try:
            _, file_extension = os.path.splitext(file.name)

            if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                return self.image_process(file)

            elif file_extension.lower() == '.pdf':
                return self.pdf_process(file)

            elif file_extension.lower() == '.txt':
                return self.text_process(file)

            elif file_extension.lower() == '.docx':
                return self.word_process(file)

            elif file_extension.lower() in ['.pptx', '.ppt']:
                return self.powerpoint_process(file)

            elif file_extension.lower() == '.mp3':
                return self.speech_process(file)

            elif file_extension.lower() == '.csv':
                return self.csv_process(file)

            elif file_extension.lower() == '.xlsx':
                return self.excel_process(file)

            else:
                raise ValueError("Unsupported file format. Please provide an image, PDF, PPT, TXT, DOCX, MP3, XLSX, or other supported file format.")
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")
            return ""

    def get_content(self, file) -> str:
        content = ""
        try:
            if file:
                file_content = self.process_file(file)
                content += file_content + "\n"
        except Exception as e:
            print(f"An error occurred while getting content: {e}")
        return content.strip()

    def get_user_input(self, file, prompt: Optional[str], messages: Optional[list], api_name: str = None) -> str:
        content = ""
        if file:
            file_content = self.get_content(file)
            if file_content:
                content += file_content + "\n"
        if prompt:
            content += prompt + "\n"
        return self.chat_with_ai(messages=messages,content=content.strip(), api_name=api_name)
