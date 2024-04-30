from utils import monthdelta, datetime, timedelta, date_validation, relativedelta

date_format = "%m/%d/%y %H:%M:%S"

# setting the given time with the clock of the server
def change_time_zone(str_date: str, timezone: int) -> str:
    date = datetime.strptime(str_date, date_format)
    date += timedelta(minutes=timezone)
    return f"{date}"


front_end_json_sample = {
'symbol': 'NQ',
'timeframe': '1min',
'chart_start_time': '2024-04-17T02:00:00Z',
'chart_end_time': '2024-04-18T15:00:00Z',
'timezone': -210,
'user_id': 10,
'history_message': [{"role": 'system', "content": '...'}, {"role": 'user', "content": '...'}, ...],
"new_message": 'Hi',
'file_path': ''
}

front_json_keys = ["symbol", "start_datetime", "end_datetime", "end_datetime", "timeframe", "lookback_days", "direction", "method", "neighborhood", "atr_coef", "min_sl_ticks", "stoploss"]

def input_filter(function_name: str, function_arguments: dict, front_json: dict):
    
    now = datetime.now()

    if function_name == "detect_trend":
        timezone = int(front_json["timezone"])
        correct_dates = True
        # validating dates
        if "start_datetime" in function_arguments or "end_datetime" in function_arguments:
            correct_dates = False
            # checking the format
            if not (date_validation(function_arguments["start_datetime"]) or date_validation(function_arguments["end_datetime"])):
                results = f"Please enter dates in the following foramat: {date_format} or specify a period of time whether for the past seconds or minutes or hours or days or weeks or years."
            # if start_datetime or end_datetime were in the future
            elif (now < datetime.strptime(function_arguments["start_datetime"], date_format)) or (now < datetime.strptime(function_arguments["end_datetime"], date_format)):
                results = "Dates should not be in the future!"
            # if end is before start
            elif datetime.strptime(function_arguments["end_datetime"], date_format) < datetime.strptime(function_arguments["start_datetime"], date_format):
                results = "End date time should be after start date time!"
            # formates are correct
            elif "lookback" in function_arguments and "start_datetime" in function_arguments:
                results = "Both lookback and datetimes could not be valued"
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
                if "symbol" not in front_json:
                    function_arguments["symbol"] = "NQ"
                else:
                    function_arguments["symbol"] = front_json["symbol"]
            if "start_datetime" not in function_arguments:
                if "start_datetime" not in front_json:
                    function_arguments["start_datetime"] = f"{now - timedelta(days=10)}"
                else:
                    function_arguments["start_datetime"] = front_json["start_datetime"]
            else:
                function_arguments["start_datetime"] = change_time_zone(function_arguments["start_datetime"], timezone)
            if "end_datetime" not in function_arguments:
                if "end_datetime" not in front_json:
                    function_arguments["end_datetime"] = f"{now}"
                else:
                    function_arguments["end_datetime"] = front_json["end_datetime"]
            else:
                function_arguments["end_datetime"] = change_time_zone(function_arguments["end_datetime"], timezone)
        
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
        if "method" not in function_arguments or function_arguments["method"] is None or function_arguments["method"] == '' or function_arguments["method"] == 'all':
            if "method" not in front_json:
                function_arguments["method"] = "nothing"
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

    else:
        raise ValueError(f"Incorrect function name: {function_name}")
    
    return function_arguments
 