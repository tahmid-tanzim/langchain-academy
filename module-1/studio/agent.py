from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode


@tool
def add(x: int, y: int) -> int:
    """
    Adds x and y.
    Args:
        x: first int
        y: second int
    """
    return x + y


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
def divide(a: int, b: int) -> float:
    """
    Divide a and b.
    Args:
        a: first int
        b: second int
    """
    return a / b


@tool
def modulo(a: float, b: int) -> float:
    """
    Remainder of a division b.
    Args:
        a: first int
        b: second int
    """
    return a % b


tools = [add, multiply, divide, modulo]

# Define LLM with bound tools
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs."
)


# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)

# Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()
