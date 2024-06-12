import single_agent.utils as utils
import logging
import re


date_format = "%b-%d-%Y %H:%M:%S"
now = utils.datetime.utcnow()


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
'history_message': [],
'new_message': '',
'file': '',
'multi-agent': False,
'user_id': '1',
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
        logging.info(f"Function name out of list: {function_name}")
    
    return function_arguments
 