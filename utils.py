import logging
import requests
import json
from openai import OpenAI
from functions_python import FunctionCalls
from functions_json import functions
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def date_validation(date_text):
    valid = True
    try:
        datetime.date.fromisoformat(date_text)
        valid = True
    except ValueError:
        valid = False
    finally:
        return valid


def monthdelta(date, delta):
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    if not m: m = 12
    d = min(date.day, [31,
        29 if y%4==0 and (not y%100==0 or y%400 == 0) else 28,
        31,30,31,30,31,31,30,31,30,31][m-1])
    return date.replace(day=d,month=m, year=y)


def get_response2(messages, functions, model, function_call):
    try:
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            functions = functions,
            function_call = function_call
        )
        return response
    except Exception as e:
        logging.info(f"Exception: {e}, Unable to generate ChatCompletion response")
        return e


def get_result2(chat_response):
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
                    if function_arguments["lookback"].split(" ")[-1] == "seconds":
                        function_arguments["start_datetime"] = f"{now - timedelta(seconds=k)}"
                    elif function_arguments["lookback"].split(" ")[-1] == "minures":
                        function_arguments["start_datetime"] = f"{now - timedelta(minutes=k)}"
                    elif function_arguments["lookback"].split(" ")[-1] == "hours":
                        function_arguments["start_datetime"] = f"{now - timedelta(hours=k)}"
                    elif function_arguments["lookback"].split(" ")[-1] == "days":
                        function_arguments["start_datetime"] = f"{now - timedelta(days=k)}"
                    elif function_arguments["lookback"].split(" ")[-1] == "weeks":
                        function_arguments["start_datetime"] = f"{now - timedelta(weeks=k)}"
                    elif function_arguments["lookback"].split(" ")[-1] == "months":
                        function_arguments["start_datetime"] = f"{monthdelta(datetime.now(), -k)}"
                    elif function_arguments["lookback"].split(" ")[-1] == "year":
                        function_arguments["start_datetime"] = f"{now - relativedelta(years=k)}"
                    else:
                        raise ValueError("???")

            # if the date formats were not valid
            if not (date_validation(function_arguments["start_datetime"]) and date_validation(function_arguments["end_datetime"])):
                results = "Please enter dates in the following foramat: YYYY-MM-DD or specify a period of time whether for the past seconds or minutes or hours or days or weeks or years."
            
            trend = FC.detect_trend(parameters=function_arguments)
            messages.append({"role": "system", "content": f"The result of the function calling with function {function_name} has become {trend}. Do not return the number which is the result of the function. Instead, use its correcsponding explanation which is in the function's description. Make sure to mention the start_datetime and end_datetime. If the user did not mention start_datetime and end_datetime tell them that they did but they should do it. Now generate a proper response."})
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
            results = f"{chat_response.choices[0].message}"
    
    return results
