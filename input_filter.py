import single_agent.utils as utils
import logging
import re


date_format = "%b-%d-%Y %H:%M:%S"
now = utils.datetime.now()


# setting the given time with the clock of the server
def change_time_zone(str_date: str, timezone: int, date_format=date_format) -> str:
    date = utils.datetime.strptime(str_date, date_format)
    date += utils.timedelta(minutes=timezone)
    return f"{date}"


def seperate_num_unit(function_arguments_lookback: str):
    if " " in function_arguments_lookback:
        lookback_number = int(function_arguments_lookback.split(" ")[0])
        lookback_unit = function_arguments_lookback.split(" ")[1]
    else:
        pattern = re.compile(r"(\d+)([a-zA-Z]+)")
        match = pattern.search(function_arguments_lookback)
        lookback_number = int(match.group(1))
        lookback_unit = match.group(2)
    return lookback_number, lookback_unit


def default_timedelta(date: str, function_arguments_lookback: str) -> str:
    if " " in function_arguments_lookback:
        lookback_number = int(function_arguments_lookback.split(" ")[0])
        lookback_unit = function_arguments_lookback.split(" ")[1]
    else:
        pattern = re.compile(r"(\d+)([a-zA-Z]+)")
        match = pattern.search(function_arguments_lookback)
        lookback_number = int(match.group(1))
        lookback_unit = match.group(2)
    
    if "." in date:
        date = date.split(".")[0]
    
    date = utils.datetime.strptime(date, date_format)
    
    if lookback_unit == "hour" or lookback_unit == "hours" or lookback_unit == "H" or lookback_unit == "h":
        return f"{(date - utils.timedelta(hours=lookback_number)).strftime(date_format)}"
    elif lookback_unit == "day" or lookback_unit == "days" or lookback_unit == "d" or lookback_unit == "D":
        return f"{(date - utils.timedelta(days=lookback_number)).strftime(date_format)}"
    elif lookback_unit == "week" or lookback_unit == "weeks" or lookback_unit == "w" or lookback_unit == "W":
        return f"{(date - utils.timedelta(weeks=lookback_number)).strftime(date_format)}"
    elif lookback_unit == "month" or lookback_unit == "months" or lookback_unit == "M" or lookback_unit == "m":
        return f"{(utils.monthdelta(date, -lookback_number)).strftime(date_format)}"
    elif lookback_unit == "year" or lookback_unit == "years" or lookback_unit == "y" or lookback_unit == "Y":
        return f"{(date - utils.relativedelta(years=lookback_number)).strftime(date_format)}"
    else:
        raise ValueError(f"\nWrong lookback unit: {lookback_unit}\n")


front_end_json_sample = {
'symbol': 'NQ',
'timeframe': '1h',
'start_datetime': 'May-1-2024 12:00:00',
'end_datetime': 'May-5-2024 12:00:00',
'timezone': -210,
'user_id': 10,
'session_id': 1,
'history_message': [],
'new_message': '',
'file': '',
'multi-agent': False,
'user_id': '1',
}



{
    'symbol': 'NQ', 
    'timeframe': '1min', 
    'start_datetime': None, 
    'end_datetime': None, 
    'timezone': -210, 
    'user_id': 24, 
    'history_message': [{'role': 'user', 'content': 'nq'}, {'role': 'system', 'content': 'To provide you with the requested information or analysis regarding the NQ, I need more specific details. Please let me know what specific information or analysis you are looking for, such as support and resistance levels, trend analysis, stop-loss calculation, take-profit calculation, or trading bias detection.'}, {'role': 'user', 'content': 'what is the trend of nq for the past week?'}, {'role': 'system', 'content': 'Based on the analysis of the past week, the trend of NQ (Nasdaq 100) is without a significant trend. This means that there is no clear upward or downward movement in the price of NQ over the past week.'}, {'role': 'user', 'content': 'what is the sr of nq for the last 10 days'}, {'role': 'system', 'content': 'The support and resistance levels for nq based on historical price data with a lookback period of 10 days and a timeframe of 1min are as follows:\n\n- Levels: [218.95, 218.66, 217.09, 218.63, 216.65, 215.96, 215.82, 215.31, 215.3, 214.9, 206.59, 207.2, 207.8446, 207.86, 208.14, 208.51, 208.6, 208.61, 208.82, 208.9, 209.07, 209.08, 209.1, 209.11, 210.22, 210.3, 210.52, 210.64, 211.02, 211.76, 211.98, 212.08, 212.55, 214.86, 214.57, 214.33, 213.44, 213.5, 212.72, 213.43, 212.89, 213.3, 213.25, 213.12, 212.93, 212.94]\n\nThese levels are determined based on historical price data and indicate areas where the price is likely to encounter support or resistance. The associated scores indicate the strength or significance of each level, with higher scores indicating stronger levels.'}, {'role': 'user', 'content': 'Calculate Support and Resistance Levels of YM by looking back up to past 10 days and timeframe of 10 minutes.'}, {'role': 'system', 'content': 'The support and resistance levels for YM based on historical price data with a lookback period of 10 days and a timeframe of 10min are as follows:\n\n- Levels: [39999.0, 39928.0, 39911.0, 39839.0, 39843.0, 39819.0, 39574.0, 39284.0, 39560.0, 39481.0]\n\nThese levels are determined based on historical price data and indicate areas where the price is likely to encounter support or resistance. The associated scores indicate the strength or significance of each level, with higher scores indicating stronger levels.'}, {'role': 'user', 'content': 'hello'}, {'role': 'system', 'content': "I'm here to help with trading and financial market queries. If you think your ask relates to trading and isn't addressed, please report a bug using the bottom right panel."}], 
    'new_message': 'heelllooowww', 
    'file': None
}


front_json_keys = ["symbol", "start_datetime", "end_datetime", "end_datetime", "timeframe", "lookback_days", "direction", "method", "neighborhood", "atr_coef", "min_sl_ticks", "stoploss"]


def input_filter(function_name: str, function_arguments: dict, front_json: dict):    

    if function_name == "detect_trend":
        timezone = int(front_json["timezone"])
        correct_dates = True
        # validating dates
        results = ""
        
        if "end_datetime" in function_arguments:            
            if function_arguments["end_datetime"] == "now":
                function_arguments["end_datetime"] = now.strftime(date_format)
        
        if ("start_datetime" in function_arguments) and ("end_datetime" not in function_arguments):
            function_arguments["end_datetime"] = now.strftime(date_format)

        if "start_datetime" in function_arguments or "end_datetime" in function_arguments:
            correct_dates = False
            # checking the format
            if not (utils.date_validation(function_arguments["start_datetime"]) or utils.date_validation(function_arguments["end_datetime"])):
                results = f"Please enter dates in the following foramat: {date_format} or specify a period of time whether for the past hours or days or weeks or years."
            # if start_datetime or end_datetime were in the future
            elif (now < utils.datetime.strptime(function_arguments["start_datetime"], date_format)) or (now < utils.datetime.strptime(function_arguments["end_datetime"], date_format)):
                results = "Dates should not be in the future!"
            # if end is before start
            elif utils.datetime.strptime(function_arguments["end_datetime"], date_format) < utils.datetime.strptime(function_arguments["start_datetime"], date_format):
                results = "End date time should be after start date time!"
            # formates are correct
            elif "lookback" in function_arguments and "start_datetime" in function_arguments:
                results = "Both lookback and datetimes could not be valued"
            # dates are valid
            else:
                correct_dates = True
        
        # if loookback and end_datetime were specified
        if ("lookback" in function_arguments and "end_datetime" in function_arguments) and correct_dates:
            function_arguments["start_datetime"] = default_timedelta(function_arguments["end_datetime"], function_arguments["lookback"])
        
        # handling the default values when none of the parameters were specified
        elif ("lookback" not in function_arguments) and correct_dates:
            if "symbol" not in function_arguments:
                if "symbol" not in front_json:
                    function_arguments["symbol"] = "NQ"
                else:
                    function_arguments["symbol"] = front_json["symbol"]
            if "start_datetime" not in function_arguments:
                if "start_datetime" not in front_json:
                    function_arguments["start_datetime"] = f"{(now - utils.timedelta(days=10)).strftime(date_format)}"
                else:
                    function_arguments["start_datetime"] = front_json["start_datetime"]
            else:
                function_arguments["start_datetime"] = change_time_zone(function_arguments["start_datetime"], timezone)
            if "end_datetime" not in function_arguments:
                if "end_datetime" not in front_json:
                    function_arguments["end_datetime"] = f"{now.strftime(date_format)}"
                else:
                    function_arguments["end_datetime"] = front_json["end_datetime"]
            else:
                function_arguments["end_datetime"] = change_time_zone(function_arguments["end_datetime"], timezone)
        
        # if just lookback was specified
        elif correct_dates:
            function_arguments["end_datetime"] = now.strftime(date_format)
            function_arguments["start_datetime"] = default_timedelta(function_arguments["end_datetime"], function_arguments["lookback"])

        return [function_arguments, results, correct_dates]

    elif function_name == "calculate_sr":
        # Handling the default values
        if "symbol" not in function_arguments:
            if "symbol" not in front_json:
                function_arguments["symbol"] = "ES"
            else:
                function_arguments["symbol"] = front_json["symbol"]
        if "timeframe" not in function_arguments:
            if "timeframe" not in front_json:
                function_arguments["timeframe"] = "1h"
            else:
                function_arguments["timeframe"] = front_json["timeframe"]
        if "lookback_days" not in function_arguments:
            if "lookback_days" not in front_json:
                function_arguments["lookback_days"] = "10 days"
            else:
                function_arguments["lookback_days"] = front_json["lookback_days"]

    elif function_name == "calculate_sl":
        # Handling the default values
        if "symbol" not in function_arguments:
            if "symbol" not in front_json:
                function_arguments["symbol"] = "NQ"
            else:
                function_arguments["symbol"] = front_json["symbol"]
        if "direction" not in function_arguments:
            if "direction" not in front_json:
                function_arguments["direction"] = 1
            else:
                function_arguments["direction"] = int(front_json["direction"])
        else:
            function_arguments["direction"] = int(function_arguments["direction"])
        if "method" not in function_arguments or function_arguments["method"] == 'all':
            if "method" not in front_json:
                function_arguments["method"] = "all"
            else:
                function_arguments["method"] = front_json["method"]
        if "neighborhood" not in function_arguments:
            if "neighborhood" not in front_json:
                function_arguments["neighborhood"] = 20
            else:
                function_arguments["neighborhood"] = int(front_json["neighborhood"])
        else:
            function_arguments["neighborhood"] = int(function_arguments["neighborhood"])
        if "atr_coef" not in function_arguments:
            if "atr_coef" not in front_json:
                function_arguments["atr_coef"] = 1.5
            else:
                function_arguments["atr_coef"] = float(front_json["atr_coef"])
        else:
            function_arguments["atr_coef"] = float(function_arguments["atr_coef"])
        if "lookback" not in function_arguments:
            if "lookback" not in front_json:
                function_arguments["lookback"] = 100
            else:
                function_arguments["lookback"] = int(front_json["lookback"])
        else:
            function_arguments["lookback"] = int(function_arguments["lookback"])
        if "min_sl_ticks" not in function_arguments:
            if "min_sl_ticks" not in front_json:
                function_arguments["min_sl_ticks"] = 4
            else:
                function_arguments["min_sl_ticks"] = int(front_json["min_sl_ticks"])
        else:
            function_arguments["min_sl_ticks"] = int(function_arguments["min_sl_ticks"])

    elif function_name == "calculate_tp":
        # Handling the default values
        if "symbol" not in function_arguments:
            if "symbol" not in front_json:
                function_arguments["symbol"] = "NQ"
            else:
                function_arguments["symbol"] = front_json["symbol"]
        if "direction" not in function_arguments:
            if "direction" not in front_json:
                function_arguments["direction"] = 1
            else:
                function_arguments["direction"] = int(front_json["direction"])
        else:
            function_arguments["direction"] = int(function_arguments["direction"])
        if "stoploss" not in function_arguments:
            if "stoploss" not in front_json:
                function_arguments["stoploss"] = 0
            else:
                function_arguments["stoploss"] = float(front_json["stoploss"])
        else:
            function_arguments["stoploss"] = float(function_arguments["stoploss"])

    elif function_name == "bias_detection":
        if "symbol" not in function_arguments:
            if "symbol" not in front_json:
                function_arguments["symbol"] = "ES"
            else:
                function_arguments["symbol"] = front_json["symbol"]
        if "method" not in function_arguments:
            if "method" not in front_json:
                function_arguments["method"] = "all"
            else:
                function_arguments["method"] = front_json["method"]

    else:
        logging.error(f"Function name out of list: {function_name}")
    
    return function_arguments
 