import functools
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage

from multi_agent.utils import create_agent, llm, agent_node, AgentState, tool_node, run_Handler, run_greeting, run_tutorial, router
from multi_agent.functions_agents import Trend, SR, SL, TP, Bias


trading_agent = create_agent(
    llm,
    [Trend, SR, TP, SL, Bias],
    system_message="""You are an assistant with the following capabilities given to you by your tools:
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
Bias: Do not mention the name of the parameters of the functions directly in the final answer. Instead, briefly explain them and use other meaningfuly related synonyms. Now generate a proper response."""
)
trading_node = functools.partial(agent_node, agent=trading_agent, name="Trading")

workflow = StateGraph(AgentState)

workflow.add_node("Trading", trading_node)
workflow.add_node("call_tool", tool_node)
workflow.add_node("Handler", run_Handler)
workflow.add_node("Greeting", run_greeting)
workflow.add_node("Tutorial", run_tutorial)

workflow.add_edge("Greeting", END)
workflow.add_edge("Tutorial", END)

workflow.add_conditional_edges(
    "Handler",
    router,
    {"continue": "Trading", "Greeting": "Greeting", "Tutorial": "Tutorial", "end": END},
)

workflow.add_conditional_edges(
    "Trading",
    router,
    {"continue": "Trading", "call_tool": "call_tool", "end": END},
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "Trading": "Trading"
    },
)
workflow.set_entry_point("Handler")

graph = workflow.compile()


def generate_multi_agent_answer(input_json):
    for s in graph.stream(
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
    ):
        if s["Trading"]["messages"][0].response_metadata["finish_reason"] == "stop":
            return s["Trading"]["messages"][0].content
