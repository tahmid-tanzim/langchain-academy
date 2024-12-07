from typing_extensions import TypedDict
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
# from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool


@tool
def plus(x: int, y: int) -> int:
    """
    Add x and y.
    Args:
        x: first int
        y: second int
    """
    return x + y


@tool
def minus(x: int, y: int) -> int:
    """
    Subtract y from x.
    Args:
        x: first int
        y: second int
    """
    return x - y


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiplies a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def modulo(a: int, b: int) -> int:
    """
    Remainder of a division b.
    Args:
        a: first int
        b: second int
    """
    return a % b


@tool
def get_small_value(a: int, b: int) -> int:
    """
    small value between a and b.
    Args:
        a: first int
        b: second int
    """
    return min(a, b)


@tool
def get_large_value(a: int, b: int) -> int:
    """
    large value between a and b.
    Args:
        a: first int
        b: second int
    """
    return max(a, b)


tools_list = [
    plus,
    minus,
    multiply,
    modulo,
    get_small_value,
    get_large_value,
]

# LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

# LLM with bound tool
llm_with_tools = llm.bind_tools(tools_list)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)

# Nodes
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools_list))

# Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)

# Compile graph
graph = builder.compile()
