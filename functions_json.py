functions = [
    
    ######### Trend Detection #########
    {
        "name": "detect_trend",
        "description": "It is designed primarily for financial data analysis and to analyzes the trend of a specified financial instrument over a given time range , enabling users to gauge the general direction of a security or market index. \
Either start_datetime along with end_datetime should be specified of lookback should be specified but both cases should not happen simultaneously. \
A number between -3 and 3 that represents the trend’s intensity and direction. The value is interpreted as follows: \
\n -3: strong bearish (downward) trend \
\n -2: moderate bearish (downward) trend \
\n -1: mild bearish (downward) trend \
\n 0: without significant trend \
\n 3: strong bullish (upward) trend \
\n 2: moderate bullish (upward) trend \
\n 1: mild bullish (upward) trend",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": ['NQ', 'ES', 'GC', 'YM', 'RTY', 'CL'],
                    "description": '''The ticker symbol of the financial instrument to be analyzed.'''
                },
                "start_datetime": {
                    "type": "string",
                    "format": "date-time",
                    "description": '''The start timestamp of period over which the analysis is done. (e.g. 3/10/2023 15:45:30)'''
                },
                "end_datetime": {
                    "type": "string",
                    "format": "date-time",
                    "description": '''The end timestamp of period over which the analysis is done. (e.g. 3/10/2023 15:45:30)'''
                },
                "lookback": {
                    "type": "string",
                    "description": '''The number of seconds, minutes, hours, days, weeks, months or years to look back for calculating the trend of the given symbol. This parameter determines the depth of historical data to be considered in the analysis. The format of this value must obey one of the following examples: 30 seconds, 10 minutes, 2 hours, 5 days, 3 weeks, 2 months and 3 years. Either start_datetime along with end_datetime should be specified of lookback should be specified but both cases should not happen simultaneously.'''
                }
            },
            "required": []
        }
    },


    ######### Calculate Support and Resistance Levels #########
    {
        "name": "calculate_sr",
        "description": # "Identifying and scoring support and resistance levels in financial markets based on historical price data. \
"Support and resistance levels are key concepts in technical analysis, representing price points on a chart where the odds favor a pause or reversal of a prevailing trend. \
This function analyzes candlestick charts over a specified timeframe and lookback period to calculate these levels and their respective strengths. \
Returns a dictionary containing four lists, each corresponding to a specific aspect of the calculated support and resistance levels: \
1. levels_prices (list of floats): The prices at which support and resistance levels have been identified. \
2. levels_start_timestamps (list of timestamps): The starting timestamps for each identified level, indicating when the level first became relevant. \
3. levels_end_timestamps (list of timestamps): The ending timestamps for each level, marking when the level was last relevant. \
4. levels_scores (list of floats): Scores associated with each level, indicating the strength or significance of the level. Higher scores typically imply stronger levels.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": ['NQ', 'ES', 'GC', 'YM', 'RTY', 'CL'],
                    "description": '''The ticker symbol of the financial instrument to be analyzed.'''
                },
                "timeframe": {
                    "type": "string",
                    "description": '''Specifies the timeframe of the candlestick chart to be analyzed. This parameter defines the granularity of the data used for calculating the levels. The only allowed formats would like 3h, 20min, 1d.'''
                },
                "lookback_days": {
                    "type": "integer",
                    "enum": ["days"],
                    "description": '''The number of days to look back for calculating the support and resistance levels. This parameter determines the depth of historical data to be considered in the analysis.'''
                }
            },
            "required": []
        }
    },    


    ######### Stop Loss Calculation #########
    {
        "name": "calculate_sl",
        "description": '''Identifying the optimal level for placing a stop-loss. Returns a dictionary same as this: \
{'sl': [17542.5], 'risk': [268.5], 'info': ['calculated based on maximum high price of previous 100 candles']} \
which includes sl value, risk on the trade and an information. \
If user don't select any method for sl calculation or select "level" method, or zigzag method the output can include \
more than one stoploss and the values type in the output can be a list such as this \
{'sl': [17542.5, 17818.25, 17858.5, 17882.5, 18518.75], 'risk': [268.5, 7.25, 47.5, 71.5, 707.75], 'info': ['minmax', 'swing', 'atr', '5min_SR', 'daily_SR']} \
It includes a list of stoplosses and the risk on them and finally the level or method name of stoploss.''',
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": ['NQ', 'ES', 'GC', 'YM', 'RTY', 'CL'],
                    "description": '''The ticker symbol of the financial instrument to be analyzed.'''
                },
                "method": {
                    "type": "string",
                    "description": '''By defalut the function calculate the SL for all methods''',
                    "enum": ["swing", "minmax", "atr", "vwap_band", "level"]
                },
                "direction": {
                    "type": "integer",
                    "enum": [1, -1],
                    "description": '''-1: means the user want to calculate stoploss for a short position. 1: means the user want to calculate stoploss for a long position'''
                },
                "lookback": {
                    "type": "integer",
                    "description": '''it is used when the method is set to 'minmax' and shows the number of candles that the SL is calculated based on them.'''
                },
                "neighborhood": {
                    "type": "integer",
                    "min": 1,
                    "max": 120,
                    "description": '''If user chooses the swing as method, they can specify the neighborhood.'''
                },
                "atr_coef": {
                    "type": "number",
                    "description": '''it is used if the method is 'atr' and shows the coefficient of atr'''
                }
            },
            "required": []
        }
    },


    ######### Take-Profit Calculation #########
    {
        "name": "calculate_tp",
        "description": '''Identifying the optimal level for placing a take-profit. Returns a list of price for take-profit and information for each price \
For exampe: {'tp': [5139.25, 5140.25, 5144.0], 'info': ['calculated based on the level VWAP_Top_Band_2', 'calculated based on the level Overnight_high', 'calculated based on the level VWAP_Top_Band_3']}''',
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": ['NQ', 'ES', 'GC', 'YM', 'RTY', 'CL'],
                    "description": '''The ticker symbol of the financial instrument to be analyzed.'''
                },
                "direction": {
                    "type": "integer",
                    "enum": [1, -1],
                    "description": '''-1: means the user want to calculate stoploss for a short position. 1: means the user want to calculate stoploss for a long position'''
                },
                "stoploss": {
                    "type": "number",
                    "description": '''the value for stoploss'''
                }
            },
            "required": []
        }
    }

]
