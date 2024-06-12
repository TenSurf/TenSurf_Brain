import json
import operator
from typing import Annotated, Sequence, TypedDict
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
        return prompt | llm.bind_functions(functions)

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

        tool_executor, _ = create_agent_tools(client=self.client, ChatWithOpenai=self.ChatWithOpenai)
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
            print(f"\nresponse: {response}\n")
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
        last_message = messages[-1]
        tool_input = json.loads(

            last_message.additional_kwargs["function_call"]["arguments"]
        )
        output_json = {}

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

        tool_executor, _ = create_agent_tools(client=self.client, ChatWithOpenai=self.ChatWithOpenai)
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


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    output_json: dict
    input_json: dict
    sender: str
