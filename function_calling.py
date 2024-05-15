import json
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import pandas as pd

import input_filter
from file_processor import FileProcessor
from functions_json import functions
from importnb import imports
with imports("ipynb"):
    import functions_python
import config
import utils


class FunctionCalling:

    def __init__(self):
        self.fileProcessor = FileProcessor()
        self.client = self.fileProcessor.client
        self.results_json = {}

    def generate_llm_response(self, messages, functions, model, function_call, temperature=0.2):
        response = self.client.chat.completions.create(
            model=model, messages=messages, functions=functions, function_call=function_call, temperature=temperature)
        return response
    
    def get_results(self, messages, chat_response, front_json=None):
        assistant_message = chat_response.choices[0].message
        messages.append(assistant_message)
        # when function calling doesn't happen
        if chat_response.choices[0].message.function_call == None:
            results = f"{chat_response.choices[0].message.content}"
        # when function calling happens
        else:
            results = ""
            # if the chat response was not none
            if assistant_message.content != None:
                results += assistant_message.content + "\n"
            # extracting the function name and its arguments
            function_name = chat_response.choices[0].message.function_call.name
            function_arguments = json.loads(chat_response.choices[0].message.function_call.arguments)
            front_json = input_filter.front_end_json_sample if front_json is None else front_json
            FC = functions_python.FunctionCalls()
            print(f"\n{chat_response.choices[0].message}\n")
            
            # Filtering Inputs
            function_arguments = input_filter.input_filter(function_name, function_arguments, front_json)
            
            if function_name == "detect_trend":
                function_arguments, results, correct_dates = function_arguments
                # results will be generated only when dates are in the correct format
                if correct_dates:
                    trend = FC.detect_trend(parameters=function_arguments)
                    messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {trend}. At any situations, never return the number which is the output of the detect_trend function. Instead, use its correcsponding explanation which is in the detect_trend function's description. Make sure to mention the start_datetime and end_datetime or the lookback parameter if the user have mentioned in their last message. If the user provide neither specified both start_datetime and end_datetime nor lookback parameters, politely tell them that they should and introduce these parameters to them so that they can use them. Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Now generate a proper response."})
                    chat_response = self.generate_llm_response(
                        messages, functions, config.azure_GPT_MODEL_3, "auto"
                    )
                    assistant_message = chat_response.choices[0].message
                    messages.append(assistant_message)
                    results += chat_response.choices[0].message.content if chat_response.choices[0].message.content != None else ""

            elif function_name == "calculate_sr":
                sr_value, sr_start_date, sr_detect_date, sr_end_date, sr_importance = FC.calculate_sr(parameters=function_arguments)
                timezone_number = int(front_json["timezone"])
                for date in sr_start_date:
                    date -= timedelta(minutes=timezone_number)
                for date in sr_end_date:
                    date -= timedelta(minutes=timezone_number)
                for date in sr_detect_date:
                    date -= timedelta(minutes=timezone_number)
                messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {sr_value} for levels_prices, {sr_start_date} for levels_start_timestamps, {sr_detect_date} for levels_detect_timestamps, {sr_end_date} for levels_end_timestamps and {sr_importance} for levels_scores with {function_arguments} as its parameters. Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Do not mention the name of the levels that the level is support or resistance. The final answer should also contain the following texts: {utils.calculate_sr_string}"})
                chat_response = self.generate_llm_response(
                    messages, functions, config.azure_GPT_MODEL_3, "auto"
                )
                results += chat_response.choices[0].message.content if chat_response.choices[0].message.content != None else ""
                self.results_json["levels_prices"] = sr_value
                self.results_json["levels_start_timestamps"] = sr_start_date
                self.results_json["levels_detect_timestamps"] = sr_detect_date
                self.results_json["levels_end_timestamps"] = sr_end_date
                self.results_json["levels_scores"] = sr_importance
                self.results_json["symbol"] = function_arguments["symbol"]

            elif function_name == "calculate_sl":
                stoploss = FC.calculate_sl(function_arguments)
                messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {stoploss}. Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. The unit of every number in the answer should be mentioned. Now generate a proper response"})
                chat_response = self.generate_llm_response(
                    messages, functions, config.azure_GPT_MODEL_3, "auto"
                )
                results += chat_response.choices[0].message.content if chat_response.choices[0].message.content != None else ""

            elif function_name == "calculate_tp":
                takeprofit = FC.calculate_tp(function_arguments)
                messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {takeprofit}. Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Now generate a proper response"})
                chat_response = self.generate_llm_response(
                    messages, functions, config.azure_GPT_MODEL_3, "auto"
                )
                results += chat_response.choices[0].message.content if chat_response.choices[0].message.content != None else ""

            elif function_name == "bias_detection":
                bias = FC.get_bias(function_arguments)
                messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {bias}. Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Now generate a proper response"})
                chat_response = self.generate_llm_response(
                    messages, functions, config.azure_GPT_MODEL_3, "auto"
                )
                results += chat_response.choices[0].message.content if chat_response.choices[0].message.content != None else ""

            elif function_name == "introduction":
                results += utils.introduction

            # elif function_name == "visualize_data":
            #     def get_and_visualize_polygon_data(ticker: str, start_datetime_str: str = None, end_datetime_str: str = None, timeframe: str = 'day', api_key: str = None, data_type: str = "realtime") -> dict:
            #         if data_type == "realtime":
            #             current_time = datetime.now()
            #             end_datetime =  current_time.strftime("%Y-%m-%dT%H:%M:%S")
            #             three_days_ago = current_time - timedelta(days=3)
            #             start_datetime = three_days_ago.strftime("%Y-%m-%dT%H:%M:%S")
            #             start_datetime_str = start_datetime
            #             end_datetime_str = end_datetime
            #             start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%dT%H:%M:%S")
            #             end_datetime = datetime.strptime(end_datetime_str, "%Y-%m-%dT%H:%M:%S")
            #             start_timestamp_ms = int(start_datetime.timestamp()) * 1000
            #             end_timestamp_ms = int(end_datetime.timestamp()) * 1000
            #             url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{start_timestamp_ms}/{end_timestamp_ms}?apiKey={api_key}"
            #         elif data_type == "historical":
            #             start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%dT%H:%M:%S")
            #             end_datetime = datetime.strptime(end_datetime_str, "%Y-%m-%dT%H:%M:%S")
            #             start_timestamp_ms = int(start_datetime.timestamp()) * 1000
            #             end_timestamp_ms = int(end_datetime.timestamp()) * 1000
            #             url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{start_timestamp_ms}/{end_timestamp_ms}?apiKey={api_key}"
            #         else:
            #             return {"error": "Invalid data type. Please specify either 'realtime' or 'historical'."}
            #         response = requests.get(url)
            #         function_response = response.json()
            #         prices = [{'time': datetime.fromtimestamp(result['t'] / 1000), 'open': result['o'], 'close': result['c'], 'high': result['h'], 'low': result['l'], 'volume': result['v']} for result in function_response['results']]
            #         data = pd.DataFrame(prices)
            #         #print(data)
            #         data.to_csv("data.csv")

            #         fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

            #         fig.add_trace(go.Candlestick(x=data['time'],
            #                         open=data['open'],
            #                         high=data['high'],
            #                         low=data['low'],
            #                         close=data['close'],
            #                         name='Candlestick Chart'), row=1, col=1)

            #         # Volume plot
            #         fig.add_trace(go.Bar(x=data['time'],
            #                 y=data['volume'],
            #                 name='Volume',
            #                 marker_color='blue'), row=2, col=1)

            #         fig.update_layout(title='Candlestick Chart with Volume for ' + ticker,
            #             yaxis_title='Price',
            #             xaxis_rangeslider_visible=False,
            #             xaxis=dict(type='category'))
            #         # save figure as an html file
            #         fig.write_html("candlestick_chart.html")

            #         #fig.show()
            #         return data
            #     polygon_api_key = "6lpCMsrDOzmm6PPpSkci73RfUvEeU9y_"
            #     ######
            #     results = get_and_visualize_polygon_data(**function_arguments, api_key=polygon_api_key)

            else:
                results += f"{chat_response.choices[0].message.content}"

        return results
    
    def generate_answer(self, messages: list, content: str, front_json: dict, file_exist):
        if not content:
            return ""
        relevance_check = self.fileProcessor.is_TunSurf_related(content)
        if "irrelevant" in relevance_check.lower():
            return "I'm here to help with trading and financial market queries. If you think your ask relates to trading and isn't addressed, please report a bug using the bottom right panel."
        # getting some complement explanation to LLM
        messages.append({"role": "system", "content": utils.complement_message})
        # prompting LLM
        messages.append({"role": "user", "content": content})
        # getting the result of the prompt
        response = self.generate_llm_response(messages, functions, config.azure_GPT_MODEL_3, "auto", 0.2)
        res = self.get_results(messages, response, front_json)
        if file_exist == 1:
            if self.fileProcessor.last_file_type == '.mp3':  # Check if the last file processed was MP3
                self.fileProcessor.text_to_speech(res, 'response.mp3')
                return res, self.results_json, 'response.mp3'
        return res, self.results_json
