from utils import monthdelta, datetime, timedelta, date_validation, relativedelta


# setting the given time with the clock of the server
def change_time_zone(front_end_json):
    if "timezone" in front_end_json:
        k = int(front_end_json["timezone"])
        s = datetime(front_end_json["start_datetime"]) - timedelta(minutes=k)
        front_end_json["start_datetime"] = f"{s}"
        e = datetime(front_end_json["end_datetime"]) - timedelta(minutes=k)
        front_end_json["end_datetime"] = f"{e}"


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

front_end_json_sample.keys()

def filter_input(front_json, function_name, function_arguments):
    for front_key in front_json.keys():
        if f"{front_key}" not in function_arguments and f"{front_key}" in front_json:
            function_arguments[f"{front_key}"] = front_json[f"{front_key}"]
    function_arguments = llm_filter_input(function_name, function_arguments)
    # timezone
    
    return function_arguments


def llm_filter_input(function_name, function_arguments):
    
    now = datetime.now()

    if function_name == "detect_trend":
        correct_dates = True
        # validating dates
        if "start_datetime" in function_arguments or "end_datetime" in function_arguments:
            correct_dates = False
            # checking the format
            if not (date_validation(function_arguments["start_datetime"]) or date_validation(function_arguments["end_datetime"])):
                results = "Please enter dates in the following foramat: %d/%m/%Y %H:%M:%S or specify a period of time whether for the past seconds or minutes or hours or days or weeks or years."
            # if start_datetime or end_datetime were in the future
            elif (now < datetime.strptime(function_arguments["start_datetime"], '%d/%m/%Y %H:%M:%S')) or (now < datetime.strptime(function_arguments["end_datetime"], '%d/%m/%Y %H:%M:%S')):
                results = "Dates should not be in the future!"
            # if end is before start
            elif datetime.strptime(function_arguments["end_datetime"], '%d/%m/%Y %H:%M:%S') < datetime.strptime(function_arguments["start_datetime"], '%d/%m/%Y %H:%M:%S'):
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
        
        return [function_arguments, results, correct_dates]

    elif function_name == "calculate_sr":
        # Handling the default values
        if "symbol" not in function_arguments:
            function_arguments["symbol"] = "ES"
        if "timeframe" not in function_arguments:
            function_arguments["timeframe"] = "1h"
        if "lookback_days" not in function_arguments:
            function_arguments["lookback_days"] = "10 days"

    elif function_name == "calculate_sl":
        # Handling the default values
        function_arguments["symbol"] = function_arguments["symbol"] if "symbol" in function_arguments else "NQ"
        function_arguments["direction"] = function_arguments["direction"] if "direction" in function_arguments else 1
        function_arguments["method"] = function_arguments["method"] if 'method' in function_arguments else 'nothing'
        if function_arguments["method"] is None or function_arguments["method"] == '' or function_arguments["method"] == 'all':
            function_arguments["method"] = 'nothing'
        function_arguments["neighborhood"] = function_arguments["neighborhood"] if 'neighborhood' in function_arguments else 20
        function_arguments["atr_coef"] = function_arguments["atr_coef"] if 'atr_coef' in function_arguments else 1.5
        function_arguments["lookback"] = function_arguments["lookback"] if 'lookback' in function_arguments else 100
        function_arguments["min_sl_ticks"] = function_arguments["min_sl_ticks"] if 'min_sl_ticks' in function_arguments else 4

    elif function_name == "calculate_tp":
        # Handling the default values
        function_arguments["symbol"] = function_arguments["symbol"] if "symbol" in function_arguments else "NQ"
        function_arguments["direction"] = function_arguments["direction"] if "direction" in function_arguments else 1
        function_arguments["stoploss"] = function_arguments["stoploss"] if "stoploss" in function_arguments else 0

    else:
        raise ValueError(f"Incorrect function name: {function_name}")
    
    return function_arguments
