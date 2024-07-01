from typing import Optional, Type
from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import ToolExecutor

import functions_python

from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.text_splitter            import CharacterTextSplitter
from langchain.embeddings               import AzureOpenAIEmbeddings
from langchain_community.vectorstores   import FAISS
from unstructured.partition.html        import partition_html
from sec_api                            import QueryApi
import requests
import json
import multi_agent.utils as utils
from gpt_researcher import GPTResearcher


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
The user can also use the words last or past before the time unit which means one. (e.g. last day means 1 day, past week means 1 week, and etc.) \
Either start_datetime along with end_datetime should be specified or lookback should be specified but both cases should not happen simultaneously.")

class CalculateTrend(BaseTool):
	name = "detect_trend"
	description = """Analyzes the trend of a specified financial instrument over a given time range. \
It is designed primarily for financial data analysis, enabling users to gauge the general direction of a security or market index. \
Whether start_datetime with end_datetime, end_datetime with lookback or lookback parameters could be valued for determining the period over which's trend wants to be detected. \
The user can also use the words last or past before the time unit which means one. \
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
	return_direct: bool = True

	def _run(
			self, symbol: str = None, timeframe: str = None, lookback_days: str = None
    ) -> dict:
		parameters = {
						"symbol": symbol,
						"timeframe": timeframe,
						"lookback_days": lookback_days
						}
		
		fc = functions_python.FunctionCalls()
		response = fc.calculate_sr(parameters)

		sr_value, sr_start_date, sr_detect_date, sr_end_date, sr_importance = response
		hard_coded_response = f"The support and resistance levels for {symbol} based on historical price data with a lookback period of {lookback_days} days and a timeframe of {timeframe} are as follows:\n\n- Levels: {sr_value}\n\nThese levels are determined based on historical price data and indicate areas where the price is likely to encounter support or resistance. The associated scores indicate the strength or significance of each level, with higher scores indicating stronger levels."

		# TODO: checking the output results
		return sr_value, sr_start_date, sr_detect_date, sr_end_date, sr_importance, hard_coded_response


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
Returns a list of price for take-profit and information for each price For exampe: \
{'tp': [5139.25, 5140.25, 5144.0], 'info': ['calculated based on the level VWAP_Top_Band_2', 'calculated based on the level Overnight_high', 'calculated based on the level VWAP_Top_Band_3']}"""

	args_schema: Type[BaseModel] = PropertiesCalculateTp
	return_direct: bool = True

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


######## LLM Researcher ########
class PropertiesLLMResearcher(BaseModel):
    query: str = Field(..., description="A random topic that will be searched on the internet")

class LLMResearcher(BaseTool):
    name = "LLMResearcher"
    description = """Useful to do research and return relevant results"""

    args_schema: Type[BaseModel] = PropertiesLLMResearcher

    def _run(
        self, query: str
    ) -> str:
      report_type = "research_report"
      researcher = GPTResearcher(query, report_type)

      return researcher.conduct_research()


######## Search The Internet ########
class PropertiesSearchTheInternet(BaseModel):
    query: str = Field(..., description="A random topic that will be searched on the internet")

class SearchTheInternet(BaseTool):
    name = "SearchTheInternet"
    description = """Useful to search the internet about a a given topic and return relevant results"""

    args_schema: Type[BaseModel] = PropertiesSearchTheInternet

    def _run(
        self, query: str
    ) -> dict:
            top_result_to_return = 4
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query})
            headers = {
                'X-API-KEY': '77976766163a61a4bae1ed8672ae79e916955783',
                'content-type': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload)
            results = response.json()['organic']
            string = []
            for result in results[:top_result_to_return]:
              try:
                string.append('\n'.join([
                    f"Title: {result['title']}", f"Link: {result['link']}",
                    f"Snippet: {result['snippet']}", "\n-----------------"
                ]))
              except KeyError:
                next

            return '\n'.join(string)


######## Search News ########
class PropertiesSearchNews(BaseModel):
    query: str = Field(..., description="A random topic that will be searched on the internet")

class SearchNews(BaseTool):
    name = "SearchNews"
    description = """Useful to search news about a company, stock or any other topic and return relevant results"""

    args_schema: Type[BaseModel] = PropertiesSearchNews

    def _run(
        self, query: str
    ) -> dict:
            top_result_to_return = 4
            url = "https://google.serper.dev/news"
            payload = json.dumps({"q": query})
            headers = {
                'X-API-KEY': '77976766163a61a4bae1ed8672ae79e916955783',
                'content-type': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload)
            results = response.json()['news']
            string = []
            for result in results[:top_result_to_return]:
              try:
                string.append('\n'.join([
                    f"Title: {result['title']}", f"Link: {result['link']}",
                    f"Snippet: {result['snippet']}", "\n-----------------"
                ]))
              except KeyError:
                next

            return '\n'.join(string)


######## 10Q ########
class PropertiesSearch10q(BaseModel):
    stock: str = Field(..., description="A random stock.")
    ask: str = Field(..., description="Question from the 10-Q of the stock")

class Search10q(BaseTool):
    name = "Search10q"
    description = """Useful to search information from the latest 10-Q form for a given stock.
                     The input to this tool should be stock and ask representing the stock ticker you are interested and what
                     question you have from it."""

    args_schema: Type[BaseModel] = PropertiesSearch10q

    def _run(
        self, stock: str, ask: str
    ) -> dict:
            queryApi = QueryApi(api_key="befd094ddadf6a9e080bf194b6d9197e753ec1d4226fc8e595ed717b2f1bc3a5")
            query = {
              "query": {
                "query_string": {
                  "query": f"ticker:{stock} AND formType:\"10-Q\""
                }
              },
              "from": "0",
              "size": "1",
              "sort": [{ "filedAt": { "order": "desc" }}]
            }

            fillings = queryApi.get_filings(query)['filings']
            if len(fillings) == 0:
              return "Sorry, I couldn't find any filling for this stock, check if the ticker is correct."
            link = fillings[0]['linkToFilingDetails']
            print(link)
            print(ask)
            answer = utils.embedding_search(link, ask)
            return answer


######## 10K ########
class PropertiesSearch10k(BaseModel):
    stock: str = Field(..., description="A random stock.")
    ask: str = Field(..., description="Question from the 10-k of the stock")

class Search10k(BaseTool):
    name = "Search10k"
    description = """Useful to search information from the latest 10-K form for a given stock.
                     The input to this tool should be stock and ask representing the stock ticker you are interested and what
                     question you have from it."""

    args_schema: Type[BaseModel] = PropertiesSearch10k

    def _run(
        self, stock: str, ask: str
    ) -> dict:
            queryApi = QueryApi(api_key="befd094ddadf6a9e080bf194b6d9197e753ec1d4226fc8e595ed717b2f1bc3a5")
            query = {
              "query": {
                "query_string": {
                  "query": f"ticker:{stock} AND formType:\"10-K\""
                }
              },
              "from": "0",
              "size": "1",
              "sort": [{ "filedAt": { "order": "desc" }}]
            }

            fillings = queryApi.get_filings(query)['filings']
            if len(fillings) == 0:
              return "Sorry, I couldn't find any filling for this stock, check if the ticker is correct."
            link = fillings[0]['linkToFilingDetails']
            answer = utils.embedding_search(link, ask)
            return answer


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
	LLM_Researcher = LLMResearcher()
	SearchInternet = SearchTheInternet()
	NewsSearch = SearchNews()
	TenQ = Search10q()
	Tenk = Search10k()
	Handler = create_irrelavant_handler(client, ChatWithOpenai)

	tools = [Trend, SR, TP, SL, Bias, Handler, SearchInternet, NewsSearch, Tenk, TenQ, LLM_Researcher]
	tool_executor = ToolExecutor(tools)
	trading_tools = [Trend, SR, SL, TP, Bias]
	researcher_tools = [SearchInternet, NewsSearch, Tenk, TenQ, LLM_Researcher]
	
	return tool_executor, trading_tools, researcher_tools
