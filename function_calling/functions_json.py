functions = [
    
    ######### Trend Detection #########
    {
        "name": "detect_trend",
        "description": "Analyzes the trend of a specified financial instrument over a given time range. \
It is designed primarily for financial data analysis, enabling users to gauge the general direction of a security or market index. \
Whether start_datetime with end_datetime, end_datetime with lookback or lookback parameters could be valued for determining the period over which's trend wants to be detected. \
The function returns a numerical value that indicates the trend intensity and direction within the specified parameters. \
Returns a number between -3 and 3 that represents the trend’s intensity and direction. The value is interpreted as follows: \
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
                    "description": '''The start timestamp of period over which the analysis is done. The format of the date should be in the following format %b-%d-%y %H:%M:%S like this example: May-1-2024 13:27:49'''
                },
                "end_datetime": {
                    "type": "string",
                    "format": "date-time",
                    "description": '''The end timestamp of period over which the analysis is done. The format of the date should be in the following format %b-%d-%y %H:%M:%S like this example: May-1-2024 13:27:49. The user can set this parameter to now. In this situation this parameter's value is the current date time.'''
                },
                "lookback": {
                    "type": "string",
                    "description": '''The number of seconds, minutes, hours, days, weeks, months or years to look back for calculating the trend of the given symbol. This parameter determines the depth of historical data to be considered in the analysis. The format of this value must obey one of the following examples: 30 seconds, 10 minutes, 2 hours, 5 days, 3 weeks, 2 months and 3 years. Either start_datetime along with end_datetime should be specified or lookback should be specified but both cases should not happen simultaneously.'''
                }
            },
            "required": []
        }
    },


    ######### Calculate Support and Resistance Levels #########
    {
        "name": "calculate_sr",
        "description": "Support and resistance levels represent price points on a chart where the odds favor a pause or reversal of a prevailing trend. \
This function analyzes candlestick charts over a specified timeframe and lookback period to calculate these levels and their respective strengths. \
Returns a dictionary containing four lists, each corresponding to a specific aspect of the calculated support and resistance levels: \
1. levels_prices (list of floats): The prices at which support and resistance levels have been identified. \
2. levels_start_timestamps (list of timestamps) \
3. levels_detect_timestamps (list of timestamps) \
4. levels_end_timestamps (list of timestamps) \
5. levels_scores (list of floats): Scores associated with each level, indicating the strength or significance of the level. Higher scores typically imply stronger levels.",
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
                    "description": '''The number of days to look back for calculating the support and resistance levels. This parameter determines the depth of historical data to be considered in the analysis. (e.g. 10 days)'''
                }
            },
            "required": []
        }
    },    


    ######### Stop Loss Calculation #########
    {
        "name": "calculate_sl",
        "description": '''Stoploss (SL) is a limitation for potential losses in a position. It's below the current price for long position and above it for short position. \
Distance between the SL and current price is named risk value. This function calculates the SL based o some different methods. \
Returns A dictionary same as this: \
{'sl': [17542.5], 'risk': [268.5], 'info': ['calculated based on maximum high price of previous 100 candles']} \
which includes sl value, risk on the trade and an information. \
If user don't select any method for sl calculation or select "level" method, or zigzag method the otput can include \
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
                    "description": '''shows the method of SL calculation''',
                    "enum": ["swing", "minmax", "atr", "DVWAP_band", "WVWAP_band", "level"]
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
                    "description": '''A parameter that is used in the swing method to define the range or window within which swings are detected. example: If the "neighborhood" parameter is set to 3, it means that the swing detection is based on considering 3 candles to the left and 3 candles to the right of the swing point.'''
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
        "description": '''Take profit (TP) is opposite of the stop-loss (SL) and is based on maximum reward that we intend to achieve from a trade. \
It represents the price level at which a trader aims to close a position to secure profits before the market reverses. \
Returns list of price for take-profit and information for each price For exampe: \
{'tp': [5139.25, 5140.25, 5144.0], 'info': ['calculated based on the level VWAP_Top_Band_2', 'calculated based on the level Overnight_high', 'calculated based on the level VWAP_Top_Band_3']} ''',
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
    },


    ######### Bias Detection #########
    {
        "name": "bias_detection",
        "description": '''Detecting trading bias through different methods or Detecting the appropriate entry point for a long or short trade. \
Returns a number between -3 and 3 that represents the trend’s intensity and direction. The value is interpreted as follows: \
-3: Strong downward , -2: downward -1: Weak downward, 0: No significant trend / Neutral, 1: Weak upward, 2.upward, 3: Strong upward''',
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
                    "description": '''The user can choose from different methods including MC, Zigzag trend, Trend detection, weekly wvap, candle stick pattern, cross ma, vp detection ,power & counter ratio.'''
                }
            },
            "required": []
        }
    },


    ######### Introduction #########
    {
        "name": "introduction",
        "description": '''If the user asks about the functionalities and capabilities of the chatbot and the list of functions that it can call, this function should be called. This function introduces the chatbot with its proficiencies and the list of functions that it can call in the cases when the user do not know about the chatbot and wants to get help so that they can use the tools which the chatbot has provided. If the user asks about the chatbot, briefly introduce yourself and your tools and functions and capabilities. Do not call this function if the user is having a greet with the chatbot.'''
    },


    # ######### Visualize Data #########
    # {
    #     "name": "visualize_data",
    #     "description": "This function retrieves real-time or historical data from the Polygon API and generates a visual representation.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "ticker": {
    #                 "type": "string",
    #                 "description": '''TThe ticker symbol of the stock, such as AAPL, representing the specific company's stock data to retrieve.'''
    #             },
    #             "timeframe": {
    #                 "type": "string",
    #                 "description": '''The timeframe of data to retrieve, such as 'day', 'hour', or 'minute', indicating the granularity of the data interval.'''
    #             },
    #             "start_datetime_str": {
    #                 "type": "string",
    #                 "format": "date-time",
    #                 "description": '''The start datetime for historical data retrieval, formatted as 'YYYY-MM-DDTHH:MM:SS', specifying the beginning timestamp of the data range.'''
    #             },
    #             "end_datetime_str": {
    #                 "type": "string",
    #                 "format": "date-time",
    #                 "description": '''The end datetime for historical data retrieval, formatted as 'YYYY-MM-DDTHH:MM:SS', indicating the end timestamp of the data range.'''
    #             },
    #             "data_type": {
    #                 "type": "string",
    #                 "enum": ["realtime", "historical"],
    #                 "description": '''Specify whether to fetch real-time or historical data. Choose from 'realtime' for current data or 'historical' for past data.'''
    #             }
    #         },
    #         "required": []
    #     }
    # }

]
