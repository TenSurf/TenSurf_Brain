from utils import monthdelta, datetime, timedelta, date_validation, relativedelta


# setting the given time with the clock of the server
def change_time_zone(function_arguments):
    if "timezone" in function_arguments:
        k = int(function_arguments["timezone"])
        s = datetime(function_arguments["start_datetime"]) - timedelta(minutes=k)
        function_arguments["start_datetime"] = f"{s}"
        e = datetime(function_arguments["end_datetime"]) - timedelta(minutes=k)
        function_arguments["end_datetime"] = f"{e}"


def frontend_filter_input(front_json, function_name, function_arguments):
    function_arguments = llm_filter_input(function_name, function_arguments)
    return function_arguments


def llm_filter_input(function_name, function_arguments):
    
    now = datetime.now()

    if function_name == "detect_trend":

        if not (date_validation(function_arguments["start_datetime"]) and date_validation(function_arguments["end_datetime"])):
            raise ValueError(f"Incorrect datetime has been entered! start_datetime: {function_arguments["start_datetime"]}, end_datetime: {function_arguments["end_datetime"]}")

        # Handling the default values
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
