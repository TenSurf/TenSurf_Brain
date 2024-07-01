import os
import json
import operator
from typing import Annotated, Sequence, TypedDict
from uuid import UUID
from openai import AzureOpenAI
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import FunctionMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt.tool_executor import ToolInvocation

from multi_agent.functions_agents import create_agent_tools
import input_filter
import config

from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.text_splitter            import CharacterTextSplitter
from langchain.embeddings               import AzureOpenAIEmbeddings
from langchain_community.vectorstores   import FAISS
from unstructured.partition.html        import partition_html
from sec_api                            import QueryApi
import requests
import json
from tokencost import count_message_tokens
from langchain_core.callbacks import BaseCallbackHandler
import typing


class Utils:
    def __init__(self, ChatWithOpenai, client):
        self.api_type = "azure"
        self.api_endpoint = 'https://tensurfbrain1.openai.azure.com/'
        self.api_version = '2023-10-01-preview'
        self.api_key = '80ddd1ad72504f2fa226755d49491a61'
        self.llm = AzureChatOpenAI(
                        api_key=self.api_key,
                        api_version=self.api_version,
                        azure_endpoint=self.api_endpoint,
                        deployment_name="gpt_35_16k",
                        temperature=0,
                        streaming=True
        )
        self.client = client
        self.ChatWithOpenai = ChatWithOpenai

    def create_agent(self, llm, tools, system_message: str):
        """Create an agent."""
        functions = [convert_to_openai_function(t) for t in tools]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    " You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    f" You have access to the following tools: {tool}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        chain = prompt | llm.bind_functions(functions)
        callbacks = [ErrorHandler()]
        chain_with_callbacks = chain.with_config(callbacks=callbacks)
        # return chain
        return chain_with_callbacks

    def agent_node(self, state, agent, name):
        result = agent.invoke(state)

        if isinstance(result, FunctionMessage):
            pass

        else:
            result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)

        return {
            "messages": [result],
            "sender": name,
            "output_json": state["output_json"]
        }

    def run_Handler(self, state):
        tool_name = "HandleIrrelevant"

        messages = state["messages"]
        last_message = messages[-1]

        # print("&"*50)
        # print(last_message.content)
        # print("&"*50)

        action = ToolInvocation(
            tool=tool_name,
            tool_input=last_message.content,
        )

        tool_executor, _, _ = create_agent_tools(client=self.client, ChatWithOpenai=self.ChatWithOpenai)
        response = tool_executor.invoke(action)
        # print(response)
        standard_response = ""
        if response == 'True':
            standard_response = "I'm here to help with trading and financial market queries. If you think your ask relates to trading and isn't addressed, please report a bug using the bottom right panel."

        function_message = FunctionMessage(
            content=standard_response, HandleIrrelevant_response=f"{str(response)}", name=action.tool
        )

        return {"messages": [function_message]}

    def run_greeting(self, state):
        greeting = self.ChatWithOpenai(system_message="You are an financial and trading assistant.",
                                # model="gpt_35_16k",
                                temperature=0,
                                max_tokens=100,
                                # client=self.client
                                )
        input = [{"role": "user", "content": "Hi"}]
        response = greeting.chat(input)

        function_message = FunctionMessage(
            content=response ,name='openai chat'
        )

        return {"messages": [function_message]}

    def run_tutorial(self, state):
        response = """I'm TenSurf Brain, your AI trading assistant within TenSurf Hub platform, designed to enhance your trading experience with advanced analytical and data-driven tools: \
1. Trend Detection: I can analyze and report the trend of financial instruments over your specified period. For example, ask me, "What is the trend of NQ stock from May-1-2024 12:00:00 until May-5-2024 12:00:00?" \
2. Support and Resistance Levels: I identify and score key price levels that may influence market behavior based on historical data. Try, "Calculate Support and Resistance Levels based on YM by looking back up to the past 10 days and a timeframe of 1 hour." \
3. Stop Loss Calculation: I determine optimal stop loss points to help you manage risk effectively. Query me like, "How much would be the optimal stop loss for a short trade on NQ?" \
4. Take Profit Calculation: I calculate the ideal exit points for securing profits before a potential trend reversal. For example, "How much would be the take-profit of a short position on Dow Jones with the stop loss of 10 points?" \
5. Trading Bias Identification: I analyze market conditions to detect the best trading biases and directions, whether for long or short positions. Ask me, "What is the current trading bias for ES?" \
Each tool is tailored to help you make smarter, faster, and more informed trading decisions. Enjoy!"""

        function_message = FunctionMessage(
            content=response ,name='hardcoded string'
        )

        return {"messages": [function_message]}

    def output_json_assigner(self, tool_name, response, symbol, input_json):
        output_json = {}
        if tool_name == "calculate_sr":
            sr_value, sr_start_date, sr_detect_date, sr_end_date, sr_importance, hard_coded_response = response
            output_json["levels_prices"] = sr_value
            output_json["levels_start_timestamps"] = sr_start_date
            output_json["levels_detect_timestamps"] = sr_detect_date
            output_json["levels_end_timestamps"] = sr_end_date
            output_json["levels_scores"] = sr_importance
            output_json["function_name"] = tool_name
            output_json["symbol"] = symbol
            output_json["timeframe"] = input_json["timeframe"]
            output_json["response"] = hard_coded_response
        if tool_name == "calculate_sl":
            output_json["stop_loss"] = response
            output_json["symbol"] = symbol
            output_json["timeframe"] = input_json["timeframe"]
        if tool_name == "calculate_tp":
            output_json["take_profit"] = response
            output_json["symbol"] = symbol
            output_json["timeframe"] = input_json["timeframe"]
        return output_json

    def tool_node(self, state):
        """This runs tools in the graph
            It takes in an agent action
            and calls that tool and
            returns the result."""
        messages = state["messages"]
        last_message = messages[-2]
        output_json = {}
        
        if "function_call" in last_message.additional_kwargs:
            tool_input = json.loads(
                last_message.additional_kwargs["function_call"]["arguments"]
            )
        else:
            last_message = state["messages"][-1]
            tool_input = json.loads(
                last_message.additional_kwargs["function_call"]["arguments"]
            )
        

        if len(tool_input) == 1 and "__arg1" in tool_input:
            tool_input = next(iter(tool_input.values()))

        tool_name = last_message.additional_kwargs["function_call"]["name"].split(".")[-1]

        action = ToolInvocation(
            tool=tool_name,
            tool_input=tool_input,
        )

        tool_input = input_filter.input_filter(tool_name, tool_input, state["input_json"])
        
        if tool_name == "detect_trend":
            tool_input, results, correct_dates = tool_input
            state["input_json"].update(tool_input)
            if not correct_dates:
                function_message = FunctionMessage(
                    content=f"{tool_name} response: {results}", name=action.tool
                )
                return {"messages": [function_message], "output_json":output_json}
        else:
            state["input_json"].update(tool_input)

        tool_executor, _, _ = create_agent_tools(client=self.client, ChatWithOpenai=self.ChatWithOpenai)
        # TODO: response for calculate_tp is an empty dictionary
        response = tool_executor.invoke(action)
        symbol = tool_input["symbol"]
        output_json = self.output_json_assigner(tool_name, response, symbol, state["input_json"])

        function_message = FunctionMessage(
            content=f"{tool_name} response: {str(response)}", name=action.tool
        )

        return {"messages": [function_message], "output_json":output_json}

    def router(self, state):
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.name != 'HandleIrrelevant':
            last_message = messages[-2]

            if "function_call" in last_message.additional_kwargs:
                return "call_tool"

            if not "function_call" in last_message.additional_kwargs and last_message.type == "function":
                return "continue"

            if not last_message.additional_kwargs and last_message.type == "human":
                return "end"

        else:
            if "Greeting" in last_message.HandleIrrelevant_response and last_message.name == 'HandleIrrelevant':
                return "Greeting"

            if "Tutorial" in last_message.HandleIrrelevant_response and last_message.name == 'HandleIrrelevant':
                return "Tutorial"

            if "True" in last_message.HandleIrrelevant_response and last_message.name == 'HandleIrrelevant':
                return "end"

            if "False" in last_message.HandleIrrelevant_response and last_message.name == 'HandleIrrelevant':
                return "continue"

            if "False" in last_message.content and last_message.name == "HandleGreeting":
                return "continue"

            if "False" not in last_message.content and last_message.name == "HandleGreeting":
                return "end"

        return "continue"
    
    def agent_router(self, state):
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.content == 'trading':
            return 'Trading'
        elif last_message.content == 'researcher':
            return 'Researcher'


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    output_json: dict
    input_json: dict
    sender: str


def download_form_html(url):
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7',
        'Cache-Control': 'max-age=0',
        'Dnt': '1',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"macOS"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)

    return response.text


def embedding_search(url, ask):
    embeddings=AzureOpenAIEmbeddings(deployment="embedding-ada-002",
                                model= "text-embedding-ada-002",
                                azure_endpoint=os.getenv("azure_api_endpoint"),
                                openai_api_type=os.getenv("openai_api_type"),
                                chunk_size=1000)
    text = download_form_html(url)
    elements = partition_html(text=text)
    content = "\n".join([str(el) for el in elements])
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 150,
        length_function = len,
        is_separator_regex = False)
    docs = text_splitter.create_documents([content])
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()
    answers = retriever.get_relevant_documents(ask, top_k=4)
    answers = "\n\n".join([a.page_content for a in answers])
    return answers


def create_supervisor():
    llm = AzureChatOpenAI(
                  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                  api_version=os.getenv("azure_api_version"),
                  azure_endpoint=os.getenv("azure_api_endpoint"),
                  deployment_name="gpt_35_16k",
                  temperature=0,
                  streaming=True
    )
    
    prompt_supervisor = """
    You are a supervisor with the task of deciding between two agents.
    Your input will be user prompts or requests, but keep in mind that you should not answer the user requests.
    Your decision will be based on whether the input string pertains to the job of a trading agent or a
    researcher agent.
    Trading Agent:
    The trading agent is an assistant equipped with the following capabilities provided by these tools:
    Calculate support and resistance levels.
    Calculate the trend of a specified financial instrument over a given time range and timeframe.
    Calculate Take Profit (TP).
    Calculate Stoploss (SL).
    Detect trading bias.
    The trading agent is not designed to answer general questions. It is the researcher's job to answer general questions.
    The trading agent only responds to requests that can be answered based on its tools.
    Researcher Agent:
    The researcher agent is an assistant equipped with the following capabilities provided by these tools:
    SearchInternet: Search the internet for relevant information.
    NewsSearch: Search news sources for recent and historical news.
    TenQ: Retrieve and analyze quarterly reports (10-Q) from SEC EDGAR.
    TenK: Retrieve and analyze annual reports (10-K) from SEC EDGAR.
    llm_Researcher: Conduct thorough research and provide detailed analysis.
    The researcher agent is designed for answering general questions such as
    'I want to buy some AAPL stock. When do you suggest I should buy it?
    Based on the descriptions of these two agent,
    you should choose between the researcher agent and the trading agent.
    Your final answer will be either 'researcher' or 'trading'.
    """

    supervisor_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            prompt_supervisor
        ),
        MessagesPlaceholder(variable_name="messages")
        ]
    )

    chain = supervisor_prompt | llm
    callbacks = [ErrorHandler()]
    chain_with_callbacks = chain.with_config(callbacks=callbacks)
    # return chain
    return chain_with_callbacks


def supervisor_node(state):
    """This will help the graph flow to decide which agent should fulfill the task."""

    messages = state["messages"]
    supervisor = create_supervisor()
    result = supervisor.invoke(messages)
    result = HumanMessage(**result.dict(exclude={"type", "name"}))

    return {
        "messages": [result],
        "sender": 'supervisor',
    }


def model_and_client_chooser(user_input, groqconnecttosurf):
    tokens = count_message_tokens(
        user_input, 
        # model='azure/gpt-4o'
        model="gpt-3.5-turbo-0613"
    )
    if tokens < 4096:
        models = groqconnecttosurf.models_low
        clients = groqconnecttosurf.clients_low
    else:
        tokens = count_message_tokens(
            user_input, 
            # model='groq/llama3-70b-8192'
            model='azure/gpt-4o'
            # model="gpt-3.5-turbo-0613"
        )
        if tokens < 8192:
            models = groqconnecttosurf.models_mid
            clients = groqconnecttosurf.clients_mid
        else:
            models = groqconnecttosurf.models_high
            clients = groqconnecttosurf.clients_high
    return models, clients, tokens


class ErrorHandler(BaseCallbackHandler):
    def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: typing.Any) -> typing.Any:
        print(f"An llm error occured: {error}")
        config.logging.error(f"An llm error occured: {error}")
        return super().on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: typing.Any) -> typing.Any:
        print(f"A tool error occured: {error}")
        config.logging.error(f"A tool error occured: {error}")
        return super().on_tool_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: typing.Any) -> typing.Any:
        print(f"A chain error occured: {error}")
        config.logging.error(f"A chain error occured: {error}")
        return super().on_chain_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_retriever_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: typing.Any) -> typing.Any:
        print(f"A retriever error occured: {error}")
        config.logging.error(f"A retriever error occured: {error}")
        return super().on_retriever_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
