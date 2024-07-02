from typing import Optional, Type
from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import ToolExecutor

import functions_python
import pytz
import datetime
from datetime import datetime, timedelta
import yfinance as yf
import input_filter


########## Yahoo finance agent #########
import datetime
current_date = datetime.datetime.now(pytz.utc).date().strftime('%Y-%m-%d')
def ndaysago(n):
  import datetime
  return (datetime.datetime.now(pytz.utc) - datetime.timedelta(days=n)).date().strftime('%Y-%m-%d')
three_days_ago = ndaysago(3)
one_year_ago = ndaysago(365)
current_year = datetime.datetime.now(pytz.utc).year

def get_next_day(input_date):
    # Convert the input date string to a datetime object
    date_obj = datetime.strptime(input_date, '%Y-%m-%d')
    # Add one day to the date
    next_day = date_obj + timedelta(days=1)
    # Format the result in the same format as the input
    next_day_str = next_day.strftime('%Y-%m-%d')
    return next_day_str

def convert_to_next_day(date_str):
    # Parse the input date string
    given_date = datetime.strptime(date_str, "%Y-%m-%d")
    # Add one day to the given date
    next_day = given_date + timedelta(days=1)
    # Format the next day as a string
    next_day_str = next_day.strftime("%Y-%m-%d")
    return next_day_str

def convert_to_utc(date_str, utctimezone_offset, startorend):
    from datetime import datetime
    # Parse the input date string
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # Set the time to midnight
    if startorend == 's':
          date_obj = date_obj.replace(hour=12, minute=0, second=0, microsecond=0)
    else:
          date_obj = date_obj.replace(hour=12, minute=0, second=0, microsecond=59)

    # Create a timedelta object for the offset
    offset = timedelta(minutes=utctimezone_offset)

    # Adjust the date and time by the offset
    if utctimezone_offset >= 0:
        utc_date_obj = date_obj - offset
    else:
        utc_date_obj = date_obj + abs(offset)

    # Get UTC date and time as strings
    utc_date_time = utc_date_obj.strftime("%Y-%m-%d %H:%M:%S")

    # Extract UTC date only
    utc_date = utc_date_obj.strftime("%Y-%m-%d")

    return utc_date

def convert_datetime_to_timezone(dataframe, offset_minutes):
    from datetime import datetime, timedelta
    if offset_minutes >= 0:
        offset = timedelta(minutes=offset_minutes)
        return dataframe.index + offset
    else:
        offset = timedelta(minutes=abs(offset_minutes))
        return dataframe.index - offset
 
def convert_to_utc_with_offset(utctimezone_offset, datetime_input):
    from datetime import datetime
    # Parse the datetime input
    dt = datetime.datetime.strptime(datetime_input, "%Y-%m-%d %H:%M:%S%z")

    # Get the UTC offset in minutes
    utc_offset_minutes = utctimezone_offset

    # Calculate the timedelta for the UTC offset
    utc_offset_timedelta = datetime.timedelta(minutes=utc_offset_minutes)

    # Apply the UTC offset
    if utc_offset_minutes >= 0:
        dt_utc = dt - utc_offset_timedelta
    else:
        dt_utc = dt + abs(utc_offset_timedelta)

    return dt_utc.strftime("%Y-%m-%d %H:%M:%S")

utctimezone_offset = input_filter.front_end_json_sample["timezone"] 

def get_historical_stock_price(ticker, start_date, end_date, interval):
    """Method to get historical stock price"""
    start_date_local = start_date
    print(start_date_local)
    end_date_local = end_date
    start_date = convert_to_utc(start_date, utctimezone_offset, 's')
    print(start_date)
    end_date = convert_to_utc(end_date, utctimezone_offset, 'e')
    import datetime
    #from datetime import datetime, timedelta
    current_date = datetime.datetime.now(pytz.utc).date().strftime('%Y-%m-%d')
    ticker_data = yf.Ticker(ticker)
    ####### start_date and end_date must be converted to utc (offset is given)
    if start_date == end_date:

          if start_date and end_date == current_date:

            data = ticker_data.history()
            def utc_convertor(dt):
                    return dt.tz_convert('UTC')
            data.index = data.index.map(utc_convertor)

            if data.empty:
              return f"There is no prices for {ticker}, please try another ticker"
            else:
              last_row = data.iloc[-1]  # Get the last row of the DataFrame
              datetime_input = str(last_row.name)
              last_row = convert_to_utc_with_offset(utctimezone_offset, datetime_input)
              dt_obj = datetime.datetime.strptime(last_row, "%Y-%m-%d %H:%M:%S")
              last_date = dt_obj.strftime("%Y-%m-%d")  # Extracts the date part
              last_time = dt_obj.strftime("%H:%M:%S")  # Extracts the time part
              last_close_price = last_row['Close']  # Get the close price from the last row
              if last_date == current_date:
                ########### test    this
                return f"The last trade of {ticker} was on {last_date} and at the time {last_time} and at the price {last_close_price}"
              else:
                last_row = data.iloc[-1]  # Get the last row of the DataFrame
                datetime_input = str(last_row.name)
                last_row = convert_to_utc_with_offset(utctimezone_offset, datetime_input)
                dt_obj = datetime.datetime.strptime(last_row, "%Y-%m-%d %H:%M:%S")
                last_date = dt_obj.strftime("%Y-%m-%d")  # Extracts the date part
                last_time = dt_obj.strftime("%H:%M:%S")  # Extracts the time part
                last_close_price = last_row['Close']  # Get the close price from the last row
                fulltzname = ticker_data.info['timeZoneFullName']
                shorttzname = ticker_data.info['timeZoneShortName']
                ########### test this
                return f"The {shorttzname}'s stock market in {fulltzname} related to the {ticker} currently is closed and the last trade of {ticker} was on {last_date} and at the time {last_time} and at the price {last_close_price}"
          else:
            from datetime import datetime
            date_object = datetime.strptime(start_date, "%Y-%m-%d")
            next_day = date_object + timedelta(days=1)
            end_date = next_day.strftime("%Y-%m-%d")
            data = ticker_data.history(start=start_date, end=get_next_day(end_date), interval=interval)
            def utc_convertor(dt):
                    return dt.tz_convert('UTC')
            data.index = data.index.map(utc_convertor)
            fulltzname = ticker_data.info['timeZoneFullName']
            shorttzname = ticker_data.info['timeZoneShortName']
            if data.empty:
              return f"On the date you want, The {shorttzname}'s stock market in {fulltzname} related to the {ticker} currently is closed, please try another date"
            else:
              price = data['Close'].iloc[0]
              return f"The last traded price of {ticker} stock on {start_date_local} is {price}"
    else:
          data = ticker_data.history(start=start_date, end=get_next_day(end_date), interval=interval)
          print('data before utc conversion: ', data)
          def utc_convertor(dt):
                    return dt.tz_convert('UTC')
          data.index = data.index.map(utc_convertor)
          if data.empty:
            from datetime import datetime
            date_objectn = datetime.strptime(start_date, "%Y-%m-%d")
            startnext_dayn = date_objectn + timedelta(days=-4)
            endnext_dayn = date_objectn + timedelta(days=-2)
            start_date1 = startnext_dayn.strftime("%Y-%m-%d")
            end_date1 = endnext_dayn.strftime("%Y-%m-%d")
            ####
            date_objectnn = datetime.strptime(end_date, "%Y-%m-%d")
            startnext_dayn = date_objectnn + timedelta(days=2)
            endnext_dayn = date_objectnn + timedelta(days=4)
            start_date2 = startnext_dayn.strftime("%Y-%m-%d")
            end_date2 = endnext_dayn.strftime("%Y-%m-%d")
            data1 = ticker_data.history(start=start_date1, end=end_date1, interval='1d')
            data2 = ticker_data.history(start=start_date2, end=end_date2, interval='1d')
            if data1.empty and data2.empty:
              return f"There is no prices for {ticker}, please try another ticker"
            else:
              fulltzname = ticker_data.info['timeZoneFullName']
              shorttzname = ticker_data.info['timeZoneShortName']
              return f"The {shorttzname}'s stock market in {fulltzname} related to the {ticker} on the requested dates was closed."
          else:
              import pandas as pd
              data_frame = data
              #import datetime
              #current_date = datetime.datetime.now().date().strftime('%Y-%m-%d')
              from datetime import datetime
              date_object = datetime.strptime(end_date, "%Y-%m-%d")
              next_day = date_object + timedelta(days=-1)
              def check_closed_dates(data_frame, start_date, end_date):
                  date_range = pd.date_range(start=start_date, end=end_date)
                  data_dates = data_frame.index.date
                  closed_dates = [date for date in date_range if date.date() not in data_dates]
                  closed_dates_df = pd.DataFrame(index=closed_dates, columns=data_frame.columns)
                  #closed_dates_str = "\n".join([str(date) for date in closed_dates])
                  if closed_dates:
                      fulltzname = ticker_data.info['timeZoneFullName']
                      shorttzname = ticker_data.info['timeZoneShortName']
                      print(f"On The following dates, the {shorttzname}'s stock market in {fulltzname} related to the {ticker} was closed, therefore there is no price data on these dates:")
                      for date in closed_dates:
                        print(date)
              check_closed_dates(data_frame, start_date, next_day.date())
              #print(closed_dates_str)
              # closed_dates_str, closed_dates_df = check_closed_dates(df, '2024-05-08', '2024-06-07')
              # print(closed_dates_str)
              # print(closed_dates_df)
              print('data after utc conversion: ', data)
              #convert again to local timezone
              dataframe = data
              dataframe.index = convert_datetime_to_timezone(dataframe, utctimezone_offset)
              print('data after local Tz conversion(paeen): ', data)

              return data

def get_stock_info(ticker):
    """Method to get informations of stock"""
    ticker_data = yf.Ticker(ticker)
    return ticker_data.info

def get_stock_news(ticker):
    """Method to get news of stock"""
    ticker_data = yf.Ticker(ticker)
    news = ticker_data.news
    out = "News: \n"
    for i in news:
      out += i["title"] + "\n"
    return out

def get_stock_earnings(ticker, start_date, end_date):
    """Method to get earnings of stock"""
    start_date_local = start_date
    end_date_local = end_date
    start_date = convert_to_utc(start_date, utctimezone_offset)
    end_date = convert_to_utc(end_date, utctimezone_offset)
    ticker_data = yf.Ticker(ticker)
    earnings = ticker_data.earnings_dates
    def is_dataframe_empty(data_frame):
      return data_frame.empty
    earnings_cleaned2 = earnings.dropna()
    if is_dataframe_empty(earnings_cleaned2):
      return f"No earning report is released by {ticker}, please request another ticker's earning report"

    if start_date == end_date:
          if start_date and end_date == 'flag':     # check if the last or latest earning report is requested
            earnings_cleaned1 = earnings.dropna()
            if is_dataframe_empty(earnings_cleaned1):
              return "On the date you want, no earning report is released, please request another date"
            else:
              return f"the last earning report released by {ticker} was on date {earnings_cleaned1.index[0].strftime('%Y-%m-%d')} and is {earnings_cleaned1['Reported EPS'].iloc[0]}"
          else:   # check earning on a specific date (maybe current date or not)
            earnings_cleaned = earnings.dropna(how = 'all')
            if is_dataframe_empty(earnings_cleaned):
              return "On the date you want, no earning report is released, please request another date"
            elif start_date in earnings_cleaned.index.strftime('%Y-%m-%d'):
              # Get the earning report for the given date
              #earning_data = earnings.loc[start_date]
              reported_eps = earnings_cleaned.loc[start_date, 'Reported EPS']
              estimate_eps = earnings_cleaned.loc[start_date, 'EPS Estimate']
              if reported_eps.notna().values[0]:
                return f"Reported EPS on {start_date_local} is {reported_eps.values[0]}."
              else:
                return f"Reported EPS on {start_date_local} is not released yet, but estimated eps is {estimate_eps.values[0]}."
            else:                                   # if requested date is not in dates of earnings
              closest_date = earnings_cleaned[earnings_cleaned.index <= start_date].index.max()
              closest_reported_eps = earnings_cleaned.loc[closest_date, 'Reported EPS']
              #print(pd.isna(closest_reported_eps))
              if pd.isna(closest_reported_eps):
                return "On the date you want, no earning report is released, please request another date"
              else:
                return f"There was no report on the date {start_date_local}, but the last report before your requested date, was on date {closest_date.date()} and is {closest_reported_eps}."
    else:
          cleaned_earnings = earnings.dropna()
          if is_dataframe_empty(cleaned_earnings):
              return "On the date you want, no earning report is released, please request another date"
          cleaned_earnings.index = pd.to_datetime(cleaned_earnings.index)
          filtered_earnings = cleaned_earnings.loc[(cleaned_earnings.index >= start_date) & (cleaned_earnings.index <= end_date)]
          def utc_convertor(dt):
                    return dt.tz_convert('UTC')
          filtered_earnings.index = filtered_earnings.index.map(utc_convertor)
          print(filtered_earnings)
          return filtered_earnings


class HistoricalStockPriceInput(BaseModel):
    f"""Inputs for get_historical_stock_price.Pay attention to these points:
    -if the user uses prompts like 'what is the historical price of AAPL at 2023-01-16 on daily timeframe?' or 'what is the historical price of AAPL on 2023-01-16 on daily timeframe?', it means that 'end_date' is equal to 'start_date' and 'start_date' is equal to 'end_date' and both of them are equal to '2023-01-16' .
    -If the user uses prompts like 'current price' or 'current stock price, 'start_date' and 'end_date' are exactly equal to {current_date}.
    -If the user uses prompts like 'last price' or 'latest price', 'start_date' and 'end_date' are exactly equal to {current_date}"""
    ticker: str = Field(description="Ticker symbol of the stock")
    start_date: str = Field(default = three_days_ago ,description= f"""This is the start date of user's query. Pay attention to these points:
    -If the user uses prompts like 'for the past month' or 'from one month ago', it means that 'start_date' is equal to {ndaysago(30)}. from two month ago, 'start_date' is equal to {ndaysago(60)}
    -If the user uses prompts like 'for the past week' or 'from one week ago', it means that 'start_date' is equal to {ndaysago(7)}. from two weeks ago, 'start_date' is equal to {ndaysago(14)}
    -If the user uses prompts like 'for the past year' or 'from one year ago', it means that 'start_date' is equal to {ndaysago(365)}. from two years ago, 'start_date' is equal to {ndaysago(730)}
    -If the user uses prompts like 'at 2024-05-01' or 'at 2024-05-01' or 'on 2024-05-01' is equal to as the same as the value of '2024-05-01' and 'end_date' is equal to 'start_date'
    -If the user uses prompts like 'current price' or 'current stock price', 'start_date' and 'end_date' are exactly equal to {current_date}.
    -If the user uses prompts like 'last price' or 'latest price', 'start_date' and 'end_date' are exactly equal to {current_date}""")
    end_date: str =   Field(default = current_date, description=f"""this is the end date of user's query. Pay attention to these points:
    -If the user uses prompts like 'at 2024-05-01' or 'at 2024-05-01' or 'on 2024-05-01', it means that 'end_date' is equal to as the same as the value of '2024-05-01' and 'start_date' is equal to 'start_date'.
    -If the user uses prompts like 'for the past month' and 'from one month ago' in her sentence, it means that , and end_date is equal to the {current_date}. always 'now' means {current_date}
    -If the user uses prompts like 'at 2024-05-01' or 'at 2024-05-01', it means that 'end_date' is equal to '2024-05-01' and 'start_date' is equal to 'end_date'
    -If the user uses prompts like 'current price' or 'current stock price', 'start_date' and 'end_date' are exactly equal to {current_date}.
    -If the user uses prompts like 'last price' or 'latest price', 'start_date' and 'end_date' are exactly equal to {current_date}""")
    interval: str = Field(default = '1d', description="interval. interval (str): The frequency of data points within the specified period. 'monthly' in user's query means '1mo', 'every one minute' in user's query means '1m', 'every two minutes' in user's query means '2m' for, 'every five minutes' in user's query means '5m', 'every fifteen minutes' in user's query means '15m', 'every thirty minutes' in user's query means '30m', 'every hour or hourly' in user's query means '60m', 'every ninety minutes' in user's query means '90m', 'every hour or hourly' in user's query means '1h', 'daily' in user's query means '1d', 'every five days' in user's query means '5d', 'weekly' in user's query means '1wk'.")

class HistoricalStockPriceTool(BaseTool):
    name = "get_historical_stock_price"
    description = f"""Useful when you want to get historical price of stock. Pay attention to these points:
        -Whenever the user uses the word 'timeframe' in his prompt, he means the same thing as 'interval'.
        -If the user uses prompts like 'current price' or 'current stock price' or 'last price', 'start_date' and 'end_date' are exactly equal to {current_date}"""

    args_schema: Type[BaseModel] = HistoricalStockPriceInput
    return_direct: bool = True
    """Whether to return the tool's output directly. Setting this to True means
       that after the tool is called, the AgentExecutor will stop looping.
    """
#You should enter the stock ticker symbol recognized by the yahoo finance, priod and interval, or star_time and end_time and interval


    def _run(self, ticker: str, start_date: str = three_days_ago, end_date: str = current_date, interval: str = '1d'):
        historical_price_response = get_historical_stock_price(ticker, start_date, end_date, interval)
        return historical_price_response

    def _arun(self, ticker: str, start_date: str = three_days_ago, end_date: str = current_date, interval: str = '1d'):
        raise NotImplementedError("get_historical_stock_price does not support async")

class GetStockInformationInput(BaseModel):
    """Inputs for get_stock_info"""
    ticker: str = Field(description="Ticker symbol of the stock")

class GetStockInformationTool(BaseTool):
    name = "get_stock_info"
    description = """
This tool is designed for retrieving information about stocks. Users should input the stock ticker symbol recognized by Yahoo Finance.

When using the 'get_stock_info' tool, the agent responds based on the following instructions:

1. **General Information Category**:
    - This category provides general details about the company.
    - Attributes included: 'address1', 'city', 'state', 'zip', 'country', 'phone', 'website', 'shortName', 'longName', 'underlyingSymbol', 'industry', 'sector', 'longBusinessSummary', 'fullTimeEmployees', 'companyOfficers'.

2. **Stock Price - Volume Information Category**:
    - This category covers information related to stock prices, trading volumes, and market performance.
    - Attributes included: 'previousClose','open','dayLow','dayHigh','regularMarketPreviousClose','regularMarketOpen','regularMarketDayLow','regularMarketDayHigh','currentPrice','targetHighPrice','targetLowPrice','targetMeanPrice','targetMedianPrice','bid','ask','bidSize','askSize','fiftyTwoWeekLow','fiftyTwoWeekHigh','fiftyDayAverage','twoHundredDayAverage','volume','regularMarketVolume','averageVolume','averageVolume10days','averageDailyVolume10Day'

3. **Financial Information Category**:
    - This category encompasses financial metrics, ratios, and other relevant financial indicators.
    - Attributes included: 'dividendRate','dividendYield','payoutRatio','beta','trailingPE','forwardPE','marketCap','priceToSalesTrailing12Months','trailingAnnualDividendRate','trailingAnnualDividendYield','currency','enterpriseValue','profitMargins','floatShares','sharesOutstanding','sharesShort','sharesShortPriorMonth','sharesShortPreviousMonthDate','dateShortInterest','sharesPercentSharesOut','heldPercentInsiders','heldPercentInstitutions','shortRatio',
        ,'shortPercentOfFloat','impliedSharesOutstanding','bookValue','priceToBook','lastFiscalYearEnd','nextFiscalYearEnd','mostRecentQuarter','earningsQuarterlyGrowth','netIncomeToCommon','trailingEps','forwardEps','pegRatio','lastDividendValue','lastDividendDate',exchange,financialCurrency,timeZoneFullName,timeZoneShortName,symbol,firstTradeDateEpochUtc,trailingPegRatio,lastSplitFactor,lastSplitDate,totalCash,totalCashPerShare,ebitda,totalDebt,quickRatio,
        ,'currentRatio','totalRevenue','debtToEquity','revenuePerShare','returnOnAssets','returnOnEquity','freeCashflow','operatingCashflow','earningsGrowth','revenueGrowth','grossMargins','ebitdaMargins','operatingMargins','numberOfAnalystOpinions','governanceEpochDate','compensationAsOfEpochDate','irWebsite'.

When the user interacts with "get_stock_info", it's important to follow these guidelines to ensure accurate responses:

- If the query pertains to specific information about a ticker, the agent provides only the corresponding parts of information as the answer. For example, if the user asks about the address of a company, the "get_stock_info" will respond with just the address without including any other unrelated information.

- If the query is about a specific category or categories of information, the agent provides exactly all corresponding parts of information within that category or categories.

- If the prompt is about general information of a ticker, the answer only includes exactly the general Information Category that mentioned above.
- If the prompt is about Stock Price - Volume Information of a ticker, the answer only includes exactly the Stock Price - Volume Information Category that mentioned above.
- If the prompt is about Financial Information of a ticker, the answer only includes exactly the Financial Information Category that mentioned above.

- If the prompt is about brief information of a ticker, the answer includes all categories but only the general information. In this case, the "get_stock_info" will provide summarized information from all three categories without going into extensive detail. This is useful for users who want a quick overview of the stock.

The agent does not respond with generic prompts like "more financial information"; instead, it provides all details in each category. This means that the "get_stock_info" does not infer additional requests for more specific information beyond what the user directly asks for. For example, if the user asks for the stock price, the "get_stock_info" will provide all relevant stock price information without waiting for further instructions.

Please ensure to adhere to these guidelines while interacting with the "get_stock_info" to receive accurate and comprehensive information about the requested stocks.
here are some examples, exactly answer based on and as the same as these examples in similar questions.
# Examples:
# 1- question: 'What is the address of Apple Inc. (AAPL)?'
#    answer: 'Apple Inc.'s address is One Apple Park Way, Cupertino, CA 95014, United States.'
#
# 2- question: 'Can you provide the stock price - volume information for Microsoft (MSFT)?'
#    answer: 'Category 2: Stock Price - Volume Information
#    - Previous Close: $247.58
#    - Open: $247.23
#    - Day Low: $244.86
#    - Day High: $248.33
#    - Regular Market Previous Close: $247.58
#    - Regular Market Open: $247.23
#    - Regular Market Day Low: $244.86
#    - Regular Market Day High: $248.33
#    - Current Price: $246.48
#    - Target High Price: $280.00
#    - Target Low Price: $230.00
#    - Target Mean Price: $253.17
#    - Target Median Price: $255.00
#    - Bid: $246.70
#    - Ask: $246.80
#    - Bid Size: 10
#    - Ask Size: 12
#    - Fifty-Two Week Low: $196.25
#    - Fifty-Two Week High: $305.84
#    - Fifty Day Average: $242.23
#    - Two Hundred Day Average: $277.84
#    - Volume: 22,580,627
#    - Regular Market Volume: 22,580,627
#    - Average Volume (10 days): 25,000,000
#    - Average Daily Volume (10 Day): 25,000,000'
#
# 3- question: 'Tell me about Alphabet Inc. (GOOGL)'
#    answer: 'Category 1: general Information
#    - Address: 1600 Amphitheatre Parkway
#    - City: Mountain View
#    - State: CA
#    - Zip: 94043
#    - Country: United States
#    - Phone: +1 650-253-0000
#    - Website: https://abc.xyz
#    - Short Name: Alphabet Inc.
#    - Long Name: Alphabet Inc.
#    - Underlying Symbol: GOOGL
#    - Industry: Internet Content & Information
#    - Sector: Communication Services
#    - Long Business Summary: Alphabet Inc. is a holding company. The Company's businesses include Google Inc. (Google) and its Internet products, such as Access, Calico, CapitalG, GV, Nest, Verily, Waymo and X. The Company's segments include Google and Other Bets.
#    - Full Time Employees: 139,995
#    - Company Officers: Sundar Pichai (CEO), Ruth Porat (CFO)'
#
# 4- question: 'Brief summary of Tesla Inc. (TSLA)'
#    answer: 'Category 1: general Information
#    - Address: 3500 Deer Creek Road
#    - City: Palo Alto
#    - State: CA
#    - Zip: 94304
#    - Country: United States
#    - Phone: +1 650-681-5000
#    - Website: https://www.tesla.com
#    - Short Name: Tesla, Inc.
#    - Long Name: Tesla, Inc.
#    - Underlying Symbol: TSLA
#    - Industry: Auto Manufacturers
#    - Sector: Consumer Cyclical
#    - Long Business Summary: Tesla, Inc. designs, develops, manufactures, leases and sells electric vehicles and energy generation and storage systems, and offer services related to its sustainable energy products.
#    - Full Time Employees: 70,757
#    - Company Officers: Elon Musk (CEO), Zachary Kirkhorn (CFO)'
#
# 5- question: 'What is the market capitalization of Amazon (AMZN)?'
#    answer: 'Amazon (AMZN)'s market capitalization is $1.82T.'
#
# 6- question: 'Can you provide more financial information for Facebook (FB)?'
#    answer: 'Category 3: Financial Information
#    - Dividend Rate: N/A
#    - Dividend Yield: N/A
#    - Payout Ratio: 0
#    - Beta: 1.36
#    - Trailing P/E: 24.48
#    - Forward P/E: 17.03
#    - Market Cap: $941.59B
#    - Price/Sales (ttm): 8.67
#    - Profit Margin: 0.34%
#    - Float Shares: 2.85B
#    - Shares Outstanding: 2.85B
#    - Shares Short: 21.94M
#    - Shares Short Prior Month: 20.53M
#    - Shares Short Previous Month Date: 2024-04-15
#    - Date Short Interest: 2024-04-15
#    - Shares Percent Shares Out: 0.0124
#    - Held Percent Insiders: 0.0041
#    - Held Percent Institutions: 0.6409
#    - Short Ratio: 0.47
#    - Short Percent Of Float: 0.0119
#    - Implied Shares Outstanding: None
#    - Book Value: $116.79
#    - Price/Book (mrq): 5.43
#    - Last Fiscal Year End: 2023-12-31
#    - Next Fiscal Year End: 2024-12-31
#    - Most Recent Quarter: 2024-03-31
#    - Earnings Quarterly Growth: 0.078
#    - Net Income To Common: $25.78B
#    - Trailing Eps: 9.01
#    - Forward Eps: 14.14
#    - PEG Ratio: 1.36
#    - Last Dividend Value: None
#    - Last Dividend Date: None
#    - Exchange: NMS
#    - Financial Currency: USD
#    - Time Zone Full Name: America/New_York
#    - Time Zone Short Name: EDT
#    - Symbol: FB
#    - First Trade Date Epoch UTC: 2012-05-18 13:00:00
#    - Trailing PEG Ratio: 1.36
#    - Last Split Factor: 1:1
#    - Last Split Date: 2016-07-25
#    - Total Cash: $77.42B
#    - Total Cash Per Share: 27.03
#    - EBITDA: $39.76B
#    - Total Debt: $13.61B
#    - Quick Ratio: 4.93
#    - Current Ratio: 4.93
#    - Total Revenue: $133.23B
#    - Debt To Equity: 12.19
#    - Revenue Per Share: 46.56
#    - Return On Assets: 0.14
#    - Return On Equity: 0.21
#    - Free Cashflow: $27.41B
#    - Operating Cashflow: $44.51B
#    - Earnings Growth: 0.132
#    - Revenue Growth: 0.328
#    - Gross Margins: 0.82
#    - EBITDA Margins: 0.298
#    - Operating Margins: 0.340
#    - Number Of Analyst Opinions: 50
#    - Governance Epoch Date: 1622524800
#    - Compensation As Of Epoch Date: 1648713600
#    - IR Website: https://investor.fb.com'

"""

    args_schema: Type[BaseModel] = GetStockInformationInput

    def _run(self, ticker: str):
        get_stock_info_response = get_stock_info(ticker)
        return get_stock_info_response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_stock_info does not support async")

class GetStockNewsInput(BaseModel):
    """Inputs for get_stock_news"""
    ticker: str = Field(description="Ticker symbol of the stock")

class GetStockNewsTool(BaseTool):
    name = "get_stock_news"
    description = """
        Useful when you want to get news about stock.
        You should enter the stock ticker symbol recognized by the yahoo finance.
        """
    args_schema: Type[BaseModel] = GetStockNewsInput

    def _run(self, ticker: str):
        get_stock_news_response = get_stock_news(ticker)
        return get_stock_news_response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_stock_news does not support async")

class GetStockEarningsInput(BaseModel):
    f"""Inputs for get_stock_earnings. if the user uses prompts like 'what is the earning report of AAPL at 2023-01-16?' or 'what is the earning report of AAPL on 2023-01-16?', it means that 'end_date' is equal to 'start_date' and 'start_date' is equal to 'end_date' and both of them are equal to '2023-01-16' .
    If the user uses prompts like 'last earning report', 'start_date' and 'end_date' are exactly equal to 'flag',
    If the user uses prompts like 'current earning report', 'start_date' and 'end_date' are exactly equal to 'flag'."""
    ticker: str = Field(description="Ticker symbol of the stock")
    start_date: str = Field(default = one_year_ago ,description=f"""start date.
    If the user uses prompts like 'for the past month' or 'from one month ago', it means that 'start_date' is equal to {ndaysago(30)}. from two month ago, 'start_date' is equal to {ndaysago(60)},
    If the user uses prompts like 'for the past week' or 'from one week ago', it means that 'start_date' is equal to {ndaysago(7)}. from two weeks ago, 'start_date' is equal to {ndaysago(14)},
    If the user uses prompts like 'for the past year' or 'from one year ago', it means that 'start_date' is equal to {ndaysago(365)}. from two years ago, 'start_date' is equal to {ndaysago(730)},
    If the user uses prompts like 'at 2024-05-01' or 'in 2024-05-01' or 'on 2024-05-01', it means that 'start_date' is equal to '2024-05-01' and 'end_date' is equal to 'start_date'.
    If the user uses prompts like 'last earning report', 'start_date' and 'end_date' are exactly equal to 'flag',
    If the user uses prompts like 'latest earning report', 'start_date' and 'end_date' are exactly equal to 'flag',
    If the user uses prompts like 'current earning report', 'end_date' and 'start_date' are exactly equal to 'flag'.""")
    end_date: str =   Field(default = current_date, description=f"""end date.
    If the user uses prompts like 'for the past month' and 'from one month ago' in her sentence, it means that , and end_date is equal to the {current_date}. always 'now' means {current_date},
    If the user uses prompts like 'at 2024-05-01' or 'on 2024-05-01' or 'in 2024-05-01', it means that 'end_date' is equal to '2024-05-01' and 'start_date' is equal to 'end_date',
    If the user uses prompts like 'last earning report' , 'end_date' and 'start_date' are exactly equal to 'flag',
    If the user uses prompts like 'latest earning report', 'end_date' and 'start_date' are exactly equal to 'flag',
    If the user uses prompts like 'current earning report', 'end_date' and 'start_date' are exactly equal to 'flag'.""")

class GetStockEarningsTool(BaseTool):
    name = "get_stock_earnings"
    description = """
        Useful when you want to get earnings of stock.
        You should enter the stock ticker symbol recognized by the yahoo finance.
        if user gives a prompt like "what is the earning report of AAPL at 2023-11-02?", 'start_date' is equal to as the same as the value of '2023-11-02' and 'end_date' is equal to 'start_date',
        If the user uses prompts like 'last earning report' or 'latest earning report', 'start_date' and 'end_date' are exactly equal to 'flag',
        If the user uses prompts like 'current earning report', 'start_date' and 'end_date' are exactly equal to 'flag'."""
    args_schema: Type[BaseModel] = GetStockEarningsInput

    return_direct: bool = True
    """Whether to return the tool's output directly. Setting this to True means
       that after the tool is called, the AgentExecutor will stop looping.
    """

    def _run(self, ticker: str, start_date: str = one_year_ago, end_date: str = current_date):
        get_stock_earnings_response = get_stock_earnings(ticker, start_date, end_date)
        return get_stock_earnings_response

    def _arun(self, ticker: str, start_date: str = one_year_ago, end_date: str = current_date):
        raise NotImplementedError("get_stock_earnings does not support async")


######## Trend Detection ########
class PropertiesCalculateTrend(BaseModel):
	symbol: Optional[str] = Field(None, description="The ticker symbol of the financial instrument to be analyzed.")

	start_datetime: Optional[str] = Field(None, description="The start timestamp of period over which the analysis is done. \
The format of the date should be in the following format %b-%d-%y %H:%M:%S like this example: May-1-2024 13:27:49")
	end_datetime: Optional[str] = Field(None, description="The end timestamp of period over which the analysis is done. \
The format of the date should be in the following format %b-%d-%y %H:%M:%S like this example: May-1-2024 13:27:49. \
The user can set this parameter to now. In this situation this parameter's value is the current date time.")

	lookback: Optional[str] = Field(None, description="The number of seconds, minutes, hours, days, weeks, months or years to look back for calculating the trend of the given symbol. \
This parameter determines the depth of historical data to be considered in the analysis. The format of this value must obey one of the following examples: 30 seconds, 10 minutes, 2 hours, 5 days, 3 weeks, 2 months and 3 years. \
Either start_datetime along with end_datetime should be specified or lookback should be specified but both cases should not happen simultaneously.")

class CalculateTrend(BaseTool):
	name = "detect_trend"
	description = """Analyzes the trend of a specified financial instrument over a given time range. \
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
\n 1: mild bullish (upward) trend"""

	args_schema: Type[BaseModel] = PropertiesCalculateTrend
	return_direct: bool = True

	def _run(
			self, symbol: str = None, start_datetime: str = None, end_datetime: str = None, lookback: str = None
    ) -> dict:

		function_arguments = {
						"symbol": symbol,
						"start_datetime": start_datetime,
						"end_datetime": end_datetime,
						"lookback": lookback
						}
		
		fc = functions_python.FunctionCalls()

		# if state
		return fc.detect_trend(function_arguments)


######## Calculate Support and Resistance ########
class PropertiesCalculateSR(BaseModel):
	symbol: Optional[str] = Field(None, description="The ticker symbol of the financial instrument to be analyzed.")
	timeframe: Optional[str] = Field(None, description="Specifies the timeframe of the candlestick chart to be analyzed. \
This parameter defines the granularity of the data used for calculating the levels. The only allowed formats would like 3h, 20min, 1d.")
	lookback_days: Optional[str] = Field(None, description="The number of days to look back for calculating the support and resistance levels. \
This parameter determines the depth of historical data to be considered in the analysis. (e.g. 10 days)")

class CalculateSR(BaseTool):
	name = "calculate_sr"
	description = """Support and resistance levels represent price points on a chart where the odds favor a pause or reversal of a prevailing trend. \
This function analyzes candlestick charts over a specified timeframe and lookback period to calculate these levels and their respective strengths. \
Returns a dictionary containing five lists, each corresponding to a specific aspect of the calculated support and resistance levels: \
1. levels_prices (list of floats): The prices at which support and resistance levels have been identified. (Each point must not be specified that whether it is support or resistance.) \
2. levels_start_timestamps (list of timestamps) \
3. levels_detect_timestamps (list of timestamps) \
4. levels_end_timestamps (list of timestamps) \
5. levels_scores (list of floats): Scores associated with each level, indicating the strength or significance of the level. Higher scores typically imply stronger levels."""

	args_schema: Type[BaseModel] = PropertiesCalculateSR

	def _run(
			self, symbol: str = None, timeframe: str = None, lookback_days: str = None
    ) -> dict:
		parameters = {
						"symbol": symbol,
						"timeframe": timeframe,
						"lookback_days": lookback_days
						}
		
		fc = functions_python.FunctionCalls()

		return fc.calculate_sr(parameters)


######## Calculate Stop Loss ########
class PropertiesCalculateSl(BaseModel):
	symbol: Optional[str] = Field(None, description="The ticker symbol of the financial instrument to be analyzed.")

	method: Optional[str] = Field(None, description="shows the method of SL calculation.")

	direction: Optional[int] = Field(None, description="-1: means the user want to calculate stoploss for a short position. 1: means the user want to calculate stoploss for a long position")

	lookback: Optional[int] = Field(None, description="it is used when the method is set to 'minmax' and shows the number of candles that the SL is calculated based on them.")

	neighborhood: Optional[int] = Field(None, description="A parameter that is used in the swing method to define the range or window within which swings are detected. example: \
If the 'neighborhood' parameter is set to 3, it means that the swing detection is based on considering 3 candles to the \
left and 3 candles to the right of the swing point.")

	atr_coef: Optional[int] = Field(None, description="it is used if the method is 'atr' and shows the coefficient of atr")

class CalculateSL(BaseTool):
	name = "calculate_sl"
	description = """Stoploss (SL) is a limitation for potential losses in a position. It's below the current price for long position and above it for short position. \
Distance between the SL and current price is named risk value. This function calculates the SL based o some different methods. \
Returns A dictionary same as this: \
{'sl': [17542.5], 'risk': [268.5], 'info': ['calculated based on maximum high price of previous 100 candles']} \
which includes sl value, risk on the trade and an information. \
If user don't select any method for sl calculation or select "level" method, or zigzag method the otput can include \
more than one stoploss and the values type in the output can be a list such as this \
{'sl': [17542.5, 17818.25, 17858.5, 17882.5, 18518.75], 'risk': [268.5, 7.25, 47.5, 71.5, 707.75], 'info': ['minmax', 'swing', 'atr', '5min_SR', 'daily_SR']} \
It includes a list of stoplosses and the risk on them and finally the level or method name of stoploss."""

	args_schema: Type[BaseModel] = PropertiesCalculateSl

	def _run(
        self, symbol: str = None, method: str = None, direction: int = None,
		lookback: int = None, neighborhood: int = None, atr_coef: int = None
    ) -> int:
		parameters = {
                      "symbol": symbol,
                      "method": method,
                      "direction": direction,
                      "lookback": lookback,
                      "neighborhood": neighborhood,
                      "atr_coef": atr_coef
                      }
		
		fc = functions_python.FunctionCalls()

		return fc.calculate_sl(parameters)


######## Calculate Take-Profit ########
class PropertiesCalculateTp(BaseModel):
	symbol: Optional[str] = Field(None, description="The ticker symbol of the financial instrument to be analyzed.")

	direction: Optional[int] = Field(None, description="-1: means the user want to calculate stoploss for a short position. 1: means the user want to calculate stoploss for a long position")

	stoploss: Optional[int] = Field(None, description="the value for stoploss")

class CalculateTp(BaseTool):
	name = "calculate_tp"
	description = """Take profit (TP) is opposite of the stop-loss (SL) and is based on maximum reward that we intend to achieve from a trade. \
It represents the price level at which a trader aims to close a position to secure profits before the market reverses. \
Returns list of price for take-profit and information for each price For exampe: \
{'tp': [5139.25, 5140.25, 5144.0], 'info': ['calculated based on the level VWAP_Top_Band_2', 'calculated based on the level Overnight_high', 'calculated based on the level VWAP_Top_Band_3']}"""

	args_schema: Type[BaseModel] = PropertiesCalculateTp

	def _run(
        self, symbol: str = None, direction: int = None, stoploss: int = None
    ) -> int:
		parameters = {
						"symbol": symbol,
						"direction": direction,
						"stoploss": stoploss
						}
		
		fc = functions_python.FunctionCalls()

		return fc.calculate_tp(parameters)


######## Bias Detection ########
class PropertiesBiasDetection(BaseModel):
	symbol: Optional[str] = Field(None, description="The ticker symbol of the financial instrument to be analyzed.")

	method: Optional[str] = Field(None, description="The user can choose from different methods including MC, Zigzag trend, \
Trend detection, weekly wvap, candle stick pattern, cross ma, vp detection ,power & counter ratio.")

class BiasDetection(BaseTool):
	name = "bias_detection"
	description = """Detecting trading bias through different methods or Detecting the appropriate entry point for a long or short trade.
Returns a number between -3 and 3 that represents the trend’s intensity and direction. The value is interpreted as follows:
-3: Strong downward , -2: downward -1: Weak downward, 0: No significant trend / Neutral, 1: Weak upward, 2.upward, 3: Strong upward"""

	args_schema: Type[BaseModel] = PropertiesBiasDetection

	def _run(
        self, symbol: str = None, method: str = None
    ) -> dict:

		parameters = {
						"symbol": symbol,
						"method": method
						}
		
		fc = functions_python.FunctionCalls()

		return fc.get_bias(parameters)


class Handlers:
	def __init__(self, client, ChatWithOpenai):
		self.client = client
		self.default_message = ["Check the following text for greeting words and do as the system message said."]
		self.ChatWithOpenai = ChatWithOpenai

	def handler_zero(self):
		handler_zero_openai = self.ChatWithOpenai(system_message="You are an assistant. Your job is to check the user input Not answer it.\
If the user input contains greeting words like ‘hello’, ‘hi’, and so on, \
then you should remove these words. \
Note that the final response is either the input with the greeting word removed, \
or ‘None’ if the input consists only of a greeting word. \
if there is no greeting word in the input, the response is the last user input itself \
without any changes. \
No other responses are possible. \
Here are some examples and the valid responses: \
Example 1 (when there is no greeting word): \
{ input: ‘What is the weather?’ response: ‘What is the weather?’ } \
Example 2 (when there is a greeting word): \
{ input: ‘Hi can you speak French?’ response: ‘Can you speak French?’ } \
Example 3 (when there is just one or more greeting words): \
{ input: ‘Hello.’ response: ‘None’ } \
Note that you are not permitted to answer user requests directly. \
Only perform the tasks instructed by system messages.",

											default_user_messages=self.default_message,
											# model="gpt_35_16k",
											temperature=0,
											max_tokens=100,
											# client=self.client
											)
		return handler_zero_openai

	def handler_one(self):
		handler_one_openai = self.ChatWithOpenai(system_message="You are an assistant. Your job is to check the user input, not answer it. \
We are a system with the name 'Tensurf Brain' or 'Tensurf'. \
If the user needs any information about us or needs a tutorial for \
how our system is working, your job is to detect these scenarios and respond with 'True'. \
If the user asks about you, the answer is also 'True'. \
Also, when the user is confused and doesn't know how to start or what to do, \
the answer is also 'True'. \
For any other input that doesn't classify in the tutorial or information \
about 'Tensurf Brain' or 'Tensurf' you should return false. \
Note that you are not permitted to answer user requests directly. \
Only perform the tasks instructed by system messages.",
											# model="gpt_35_16k",
											temperature=0,
											max_tokens=100,
											# client=self.client
											)
		return handler_one_openai

	def handler_two(self):
		handler_two_openai = self.ChatWithOpenai(system_message="You are an assistant. Your job is to check the user input. \
If the user input does not contain any financial and \
trading topics or requests, then you should answer only\
with ‘True’. Otherwise, return ‘False’. \
Possible responses for you are 'True' or 'False'. \
For example, the correct response to the question \
'What is the trend?' is 'False'.",
									# model="gpt_35_16k",
									temperature=0,
									max_tokens=100,
									# client=self.client
									)
		return handler_two_openai


def create_irrelavant_handler(client, ChatWithOpenai):
	######## Irrelevant Handler ########
	class HandleIrrelevantSchema(BaseModel):
		massage: str = Field(..., description="The humanmassage")

	class HandleIrrelevant(BaseTool):
		name = "HandleIrrelevant"
		description = """This function checks if the message contains financial or trading subjects or not. \
The output of this function is either True or False and the possible output of this function are 'True' or 'False'.: \
True: when the message contains financial or trading subjects. \
False: when the message request is not in these fields."""

		args_schema: Type[BaseModel] = HandleIrrelevantSchema

		def _run(
			self, massage: str
		) -> dict:

			handlers = Handlers(client=client, ChatWithOpenai=ChatWithOpenai)
			user_input = [{"role": "user", "content": "'" + massage + "'"}]

			handler_zero_openai = handlers.handler_zero()
			handler_one_openai = handlers.handler_one()
			handler_two_openai = handlers.handler_two()

			modified_input = handler_zero_openai.chat(user_input)
			if modified_input == 'None':
				return 'Greeting'

			modified_user_input = [{"role": "user", "content": modified_input}]

			if handler_one_openai.chat(modified_user_input) == 'True':
				return "Tutorial"
			else:
				return handler_two_openai.chat(modified_user_input)
	
	Handler = HandleIrrelevant()
	return Handler


def create_agent_tools(client, ChatWithOpenai):

	Trend = CalculateTrend()
	SR = CalculateSR()
	SL = CalculateSL()
	TP = CalculateTp()
	Bias = BiasDetection()
	Handler = create_irrelavant_handler(client, ChatWithOpenai)
	hist = HistoricalStockPriceTool()
	info = GetStockInformationTool()
	news = GetStockNewsTool()
	earnings = GetStockEarningsTool()

	#tools = [Trend, SR, TP, SL, Bias, Handler]
	tools = [Trend, SR, TP, SL, Bias, Handler, hist, info, news, earnings]
	tool_executor = ToolExecutor(tools)
	trading_tools = [Trend, SR, SL, TP, Bias, hist, info, news, earnings]
	#yfinance_tools = [hist, info, news, earnings]
	


	return tool_executor, trading_tools
