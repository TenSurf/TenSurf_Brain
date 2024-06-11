import functools
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage

from multi_agent.utils import Utils, AgentState, supervisor_node
from multi_agent.functions_agents import create_agent_tools


class Multi_Agent:
    def __init__(self, ChatWithOpenai, client):
        self.output_json = {}
        self.ChatWithOpenai = ChatWithOpenai
        self.client = client
        self.utils = Utils(self.ChatWithOpenai, self.client)

    def initialize_graph(self):
        _, trading_tools_list, researcher_tools_list = create_agent_tools(client=self.client, ChatWithOpenai=self.ChatWithOpenai)
        trading_agent = self.utils.create_agent(
            self.utils.llm,
            trading_tools_list,
            system_message="""You are an assistant with the following capabilities given to you by your tools: \
SR: Calculate support and resistance levels. \
Trend: Calculate the trend of a specified financial instrument over a given time range and timeframe. \
TP: Calculate Take profit (TP), \
SL: Calculate Stoploss (SL), \
Bias: Detecting trading bias. \
Note the following rules for result of each tool: \
SR:Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Do not mention the name of the levels that the level is support or resistance. The final answer should also contain the following texts: These levels are determined based on historical price data and indicate areas where the price is likely to encounter support or resistance. The associated scores indicate the strength or significance of each level, with higher scores indicating stronger levels. \
Trend:At any situations, never return the number which is the output of the Trend function. Instead, use its correcsponding explanation which is in the Trend tool's description. Make sure to mention the start_datetime and end_datetime or the lookback parameter if the user have mentioned in their last message. If the user provide neither specified both start_datetime and end_datetime nor lookback parameters, politely tell them that they should and introduce these parameters to them so that they can use them. Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Now generate a proper response. \
TP: Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Now generate a proper response. \
SL: Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. The unit of every number in the answer should be mentioned. Now generate a proper response. \
Bias: Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Now generate a proper response. \
And Do not mention 'FINAL ANSWER' in final asnwer."""
        )
        trading_node = functools.partial(self.utils.agent_node, agent=trading_agent, name="Trading")

        researcher_agent = self.utils.create_agent(
            self.utils.llm,
            researcher_tools_list,
            system_message="""You are an assistant with the following capabilities given to you by your tools: \
SearchInternet: Search the internet for relevant information. \
NewsSearch: Search news sources for recent and historical news. \
TenQ: Retrieve and analyze quarterly reports (10-Q) from SEC EDGAR. \
TenK: Retrieve and analyze annual reports (10-K) from SEC EDGAR. \
Researcher: Conduct thorough research and provide detailed analysis. \
Your job is to answer the user's request and prompt based on your tools. You should not answer the user without using your tools.\
You are the agent following the trading agent, so you will likely see some messages inside 'state.' Ignore what happened before you and act independently to answer the user's request. \
Your primary objective is to research SEC EDGAR filings for stocks and provide comprehensive investment analysis. \
you can use the SearchInternet, NewsSearch, TenQ, TenK, and Researcher tools multiple times. \
Ensure your analysis includes financial health, market position, recent developments, and potential investment risks and opportunities. \
"""
)
        
        researcher_node = functools.partial(self.utils.agent_node, agent=researcher_agent, name="Researcher")

        workflow = StateGraph(AgentState)

        workflow.add_node("Trading", trading_node)
        workflow.add_node("Researcher", researcher_node)
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("call_tool", self.utils.tool_node)
        workflow.add_node("Handler", self.utils.run_Handler)
        workflow.add_node("Greeting", self.utils.run_greeting)
        workflow.add_node("Tutorial", self.utils.run_tutorial)

        workflow.add_edge("Greeting", END)
        workflow.add_edge("Tutorial", END)

        workflow.add_conditional_edges(
            "Handler",
            self.utils.router,
            {"continue": "supervisor", "Greeting": "Greeting", "Tutorial": "Tutorial", "end": END},
        )

        workflow.add_conditional_edges(
            "supervisor",
            self.utils.agent_router,
            {"Trading": "Trading", "Researcher": "Researcher"},
        )

        workflow.add_conditional_edges(
            "Trading",
            self.utils.router,
            {"continue": "Trading", "call_tool": "call_tool", "end": END},
        )

        workflow.add_conditional_edges(
            "Researcher",
            self.utils.router,
            {"continue": "Researcher", "call_tool": "call_tool", "end": END},
        )

        workflow.add_conditional_edges(
            "call_tool",
            lambda x: x["sender"],
            {
                "Trading": "Trading",
                "Researcher": "Researcher"
            },
        )
        workflow.set_entry_point("Handler")

        graph = workflow.compile()

        return graph

    def generate_multi_agent_answer(self, input_json, graph):
        generated_messages = graph.stream(
            {
                "messages": [
                    HumanMessage(
                        content=input_json["new_message"]
                    )
                ],
                "input_json": input_json
            },
            # Maximum number of steps to take in the graph
            {"recursion_limit": 150},
        )
        
        generated_messages = list(generated_messages)
        k = list(generated_messages[-1].keys())[0]
        if "output_json" in generated_messages[-1][k]:
            self.output_json = {
                "response": generated_messages[-1][k]["messages"][0].content,
                "chart_info": generated_messages[-1][k]["output_json"]
            }
        else:
            self.output_json = {
                "response": generated_messages[-1][k]["messages"][0].content
            }
        
        return self.output_json
