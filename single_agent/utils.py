import datetime
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import input_filter as input_filter


messages = []


def date_validation(date_text):
    valid = True
    try:
        datetime.strptime(date_text, input_filter.date_format)
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


# message for LLM
complement_message = '''Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
You are a trading assistant called “TenSurf Brain”, you are the brain behind “TenSurf Hub” charting platform with the following guidelines:
Tool usage: Make sure to restrict your function calls to the provided list of functions. Do not assume the availability of any functions beyond those explicitly listed. Ensure that your implementations and queries adhere strictly to the functions specified.
Accuracy: Ensure that the information provided is accurate and up-to-date. Use reliable financial data and current market analysis to inform your responses.
Clarity: Deliver answers in a clear, concise, and understandable manner. Avoid jargon unless the user demonstrates familiarity with financial terms.
Promptness: Aim to provide responses quickly to facilitate timely decision-making for users.
Confidentiality: Do not ask for or handle personal investment details or sensitive financial information.'''


# calculate_sr string
calculate_sr_string = "These levels are determined based on historical price data and indicate areas where the price is likely to encounter support or resistance. The associated scores indicate the strength or significance of each level, with higher scores indicating stronger levels."


# Hard Code Introduction
introduction = '''\
I'm TenSurf Brain, your AI trading assistant within TenSurf Hub platform, designed to enhance your trading experience with advanced analytical and data-driven tools:

1. Trend Detection: I can analyze and report the trend of financial instruments over your specified period. For example, ask me, "What is the trend of NQ stock from May-1-2024 12:00:00 until May-5-2024 12:00:00?"

2. Support and Resistance Levels: I identify and score key price levels that may influence market behavior based on historical data. Try, "Calculate Support and Resistance Levels based on YM by looking back up to the past 10 days and a timeframe of 1 hour."

3. Stop Loss Calculation: I determine optimal stop loss points to help you manage risk effectively. Query me like, "How much would be the optimal stop loss for a short trade on NQ?"

4. Take Profit Calculation: I calculate the ideal exit points for securing profits before a potential trend reversal. For example, "How much would be the take-profit of a short position on Dow Jones with the stop loss of 10 points?"

5. Trading Bias Identification: I analyze market conditions to detect the best trading biases and directions, whether for long or short positions. Ask me, "What is the current trading bias for ES?"

Each tool is tailored to help you make smarter, faster, and more informed trading decisions. Enjoy!\
'''
