from openai import OpenAI
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
from openai import AzureOpenAI

from functions_json import functions
from importnb import imports
with imports("ipynb"):
    import functions_python
import config
from datetime import timezone
from utils import *


class FileProcessor:
    def __init__(self):
        self.openai_api_key = config.openai_api_key
        self.client = AzureOpenAI(
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_api_endpoint
            )

    def chat_with_ai(self, messages: list, content: str):
        if not content:
            return ""
        
        def get_response(messages, functions, model, function_call, temperature=0.2):
            response = self.client.chat.completions.create(
                model=model, messages=messages, functions=functions, function_call=function_call, temperature=temperature)
            return response
        
        def get_result(messages, chat_response):
            assistant_message = chat_response.choices[0].message
            messages.append(assistant_message)
            
            if chat_response.choices[0].message.function_call == None:
                results = f"{chat_response.choices[0].message.content}"
            
            else:
                results = ""
                if assistant_message.content != None:
                    results += assistant_message.content + "\n"
                function_name = chat_response.choices[0].message.function_call.name
                function_arguments = json.loads(chat_response.choices[0].message.function_call.arguments)
                FC = functions_python.FunctionCalls()
                print(f"\n{chat_response.choices[0].message}\n")
                now = functions_python.datetime.now()

                if function_name == "detect_trend":
                    correct_dates = True
                    # validating dates
                    if "start_datetime" in function_arguments or "end_datetime" in function_arguments:
                        correct_dates = False
                        # checking the format
                        if not (date_validation(function_arguments["start_datetime"]) or date_validation(function_arguments["end_datetime"])):
                            results += "Please enter dates in the following foramat: %d/%m/%Y %H:%M:%S or specify a period of time whether for the past seconds or minutes or hours or days or weeks or years."
                        # if start_datetime or end_datetime were in the future
                        elif (now < datetime.strptime(function_arguments["start_datetime"], '%d/%m/%Y %H:%M:%S')) or (now < datetime.strptime(function_arguments["end_datetime"], '%d/%m/%Y %H:%M:%S')):
                            results += "Dates should not be in the future!"
                        # if end is before start
                        elif datetime.strptime(function_arguments["end_datetime"], '%d/%m/%Y %H:%M:%S') < datetime.strptime(function_arguments["start_datetime"], '%d/%m/%Y %H:%M:%S'):
                            results += "End date time should be after start date time!"
                        # formates are correct
                        elif "lookback" in function_arguments and "start_datetime" in function_arguments:
                            results += "Both lookback and datetimes could not be valued"
                        # dates are valid
                        else:
                            correct_dates = True
                    
                    # if loookback and end_datetime were specified
                    if ("lookback" in function_arguments and "end_datetime" in function_arguments) and correct_dates:
                        end_datetime = datetime.strptime(function_arguments["end_datetime"], '%d/%m/%Y %H:%M:%S')
                        k = int(function_arguments["lookback"].split(" ")[0])
                        function_arguments["start_datetime"] = f"{end_datetime - timedelta(days=k)}"
                    
                    # handling the default values when none of the parameters were specified
                    elif ("lookback" not in function_arguments) and correct_dates:
                        if "symbol" not in function_arguments:
                            function_arguments["symbol"] = "NQ"
                        if "start_datetime" not in function_arguments:
                            function_arguments["start_datetime"] = f"{now - timedelta(days=10)}"
                        if "end_datetime" not in function_arguments:
                            function_arguments["end_datetime"] = f"{now}"
                    
                    # if just lookback was specified
                    elif correct_dates:
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
                            raise ValueError("wrong value of time")

                    # results will be generated only when dates are in the correct format
                    if correct_dates:
                        trend = FC.detect_trend(parameters=function_arguments)
                        messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {trend}. At any situations, never return the number which is the output of the detect_trend function. Instead, use its correcsponding explanation which is in the detect_trend function's description. Make sure to mention the start_datetime and end_datetime. If the user provide neither specified both start_datetime and end_datetime nor lookback parameters, politely tell them that they should and introduce these parameters to them so that they can use them. Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Now generate a proper response."})
                        chat_response = get_response(
                            messages, functions, config.azure_GPT_MODEL_3, "auto"
                        )
                        assistant_message = chat_response.choices[0].message
                        messages.append(assistant_message)
                        results += chat_response.choices[0].message.content

                elif function_name == "calculate_sr":
                    # correcting function_arguments
                    if "symbol" not in function_arguments:
                        function_arguments["symbol"] = "ES"
                    if "timeframe" not in function_arguments:
                        function_arguments["timeframe"] = "1h"
                    if "lookback_days" not in function_arguments:
                        function_arguments["lookback_days"] = "10 days"

                    sr_value, sr_start_date, sr_end_date, sr_importance = FC.calculate_sr(parameters=function_arguments)
                    messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {sr_value} for levels_prices, {sr_start_date} for levels_start_timestamps, {sr_end_date} for levels_end_timestamps and {sr_importance} for levels_scores. Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. If the user didn't specified lookback_days or timeframe parameters, introduce these parameters to them so that they can use these parameters. Now generate a proper response"})
                    chat_response = get_response(
                        messages, functions, config.azure_GPT_MODEL_3, "auto"
                    )
                    results += chat_response.choices[0].message.content

                elif function_name == "calculate_sl":
                    stoploss = FC.calculate_sl(function_arguments)
                    messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {stoploss}. Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. The unit of every number in the answer should be mentioned. Now generate a proper response"})
                    chat_response = get_response(
                        messages, functions, config.azure_GPT_MODEL_3, "auto"
                    )
                    results += chat_response.choices[0].message.content

                elif function_name == "calculate_tp":
                    takeprofit = FC.calculate_tp(function_arguments)
                    messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {takeprofit}. Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Now generate a proper response"})
                    chat_response = get_response(
                        messages, functions, config.azure_GPT_MODEL_3, "auto"
                    )
                    results += chat_response.choices[0].message.content

                else:
                    raise ValueError(f"{chat_response.choices[0].message}")
                
            return results
        
        messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
        messages.append({"role": "user", "content": content})
        response = get_response(messages, functions, config.azure_GPT_MODEL_3, "auto")
        res = get_result(messages, response)
        return res

    def image_process(self, file_path: str) -> str:
        try:
            print("Image file detected. VisionGPT processing...")
            # OpenAI API Key
            # Perform VisionGPT processing here
            def encode_image(file_path):
                with open(file_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')

            image_path = file_path
            base64_image = encode_image(image_path)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
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

    def pdf_process(self, file_path: str) -> str:
        try:
            print("PDF file detected. Extracting text...")
            with open(file_path, 'rb') as file:
                pdf = PyPDF2.PdfReader(file)
                file_contents = ''
                for page in range(len(pdf.pages)):
                    file_contents += pdf.pages[page].extract_text()
            print("Extraction complete.")
            return file_contents
        except Exception as e:
            print(f"An error occurred while processing the PDF: {e}")
            return ""

    def text_process(self, file_path: str) -> str:
        try:
            print("Text file detected. Extracting text...")
            with open(file_path, 'r') as file:
                file_contents = file.read()
            print("Extraction complete.")
            return file_contents
        except Exception as e:
            print(f"An error occurred while processing the text file: {e}")
            return ""

    def word_process(self, file_path: str) -> str:
        try:
            print("Word document detected. Extracting text...")
            doc = docx.Document(file_path)
            file_contents = '\n'.join([para.text for para in doc.paragraphs])
            print("Extraction complete.")
            return file_contents
        except Exception as e:
            print(f"An error occurred while processing the Word document: {e}")
            return ""

    def powerpoint_process(self, file_path: str) -> str:
        try:
            print("Powerpoint document detected. Extracting text...")
            # Load the PowerPoint presentation
            prs = Presentation(file_path)
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

    def speech_process(self, file_path: str) -> str:
        try:
            print("Speech file detected. Processing speech...")
            os.environ['OPENAI_API_KEY'] = self.openai_api_key
            speech= open(file_path, "rb")
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file= speech)
            speech_contents = transcription.text
            print("Speech processing complete.")
            return speech_contents
        except Exception as e:
            print(f"An error occurred while processing the speech file: {e}")
            return ""

    def excel_process(self, file_path: str) -> str:
        try:
            print("Excel file detected. Extracting text...")
            wb = load_workbook(file_path)
            file_contents = '\n'.join([str(cell.value) for sheet in wb.sheetnames for row in wb[sheet].iter_rows() for cell in row if cell.value is not None])
            print("Extraction complete.")
            return file_contents
        except Exception as e:
            print(f"An error occurred while processing the Excel file: {e}")
            return ""

    def csv_process(self, file_path: str) -> str:
        try:
            print("CSV file detected. Extracting text...")
            with open(file_path, 'r', newline='', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                file_contents = '\n'.join(','.join(row) for row in csv_reader)
            print("Extraction complete.")
            return file_contents
        except Exception as e:
            print(f"An error occurred while processing the CSV file: {e}")
            return ""

    def process_file(self, file_path: str) -> str:
        try:
            _, file_extension = os.path.splitext(file_path)

            if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                return self.image_process(file_path)

            elif file_extension.lower() == '.pdf':
                return self.pdf_process(file_path)

            elif file_extension.lower() == '.txt':
                return self.text_process(file_path)

            elif file_extension.lower() == '.docx':
                return self.word_process(file_path)

            elif file_extension.lower() in ['.pptx', '.ppt']:
                return self.powerpoint_process(file_path)

            elif file_extension.lower() == '.mp3':
                return self.speech_process(file_path)

            elif file_extension.lower() == '.csv':
                return self.csv_process(file_path)

            elif file_extension.lower() == '.xlsx':
                return self.excel_process(file_path)

            else:
                raise ValueError("Unsupported file format. Please provide an image, PDF, PPT, TXT, DOCX, MP3, XLSX, or other supported file format.")
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")
            return ""

    def get_content(self, file_path: Optional[str]) -> str:
        content = ""
        try:
            if file_path:
                file_content = self.process_file(file_path)
                content += file_content + "\n"
        except Exception as e:
            print(f"An error occurred while getting content: {e}")
        return content.strip()

    def get_user_input(self, file_path: Optional[str], prompt: Optional[str], messages: Optional[list]) -> str:
        content = ""
        if file_path:
            file_content = self.get_content(file_path)
            if file_content:
                content += file_content + "\n"
        if prompt:
            content += prompt + "\n"
        return self.chat_with_ai(messages=messages,content=content.strip())


# # For Debugging
# prompts = [
    # # detect_trend
    # "What is the trend of NQ stock from 20/4/2024 15:45:30 until 24/4/2024 15:45:30?",
    # # calculate_sr
    # "Calculate Support and Resistance Levels based on ES by looking back up to past 10 days and timeframe of 1 hour.",
    # # calculate_sl
    # "How much would be the stop loss for trading based on NQ and short positions with minmax method by looking back up to 30 candles and considering 50 candles neighboring the current time and also attribute coefficient of 1.3?",
    # # calculate_tp
    # "How much would be the take-profit of the NQ with the stop loss of 10 and direction of 1?"
    # # parallel function calling
    # "detect the trend of NQ and calculate sr of it"
# ]

# # saving the answer of the prompt in a dictionary which its key is the prompt and its value is the answer to that prompt
# results = {}
# from utils import messages

# getting the answer of the prompts
# try:
# llm = FileProcessor()
# for prompt in prompts:
#     result = llm.get_user_input(file_path=None, prompt=prompt, messages=messages)
#     print(f"{prompt}    =>    {result}")
#     results[prompt]=result

# except Exception as e:
#     print(f"The following exception occured:\n{e}")

# finally:
#     # saving the answers
#     with open("test_results.json", "w") as outfile: 
#         json.dump(results, outfile)
