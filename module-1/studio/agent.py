from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel, Field


# def add(a: int, b: int) -> int:
#     """Adds a and b.
#
#     Args:
#         a: first int
#         b: second int
#     """
#     return a + b

class Add(BaseModel):
    """Adds two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

# def multiply(a: int, b: int) -> int:
#     """Multiplies a and b.
#
#     Args:
#         a: first int
#         b: second int
#     """
#     return a * b


class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


# def divide(a: int, b: int) -> float:
#     """Divide a and b.
#
#     Args:
#         a: first int
#         b: second int
#     """
#     return a / b


class Divide(BaseModel):
    """Divide two integers a and b."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


# tools = [add, multiply, divide]
tools = [Add, Multiply, Divide]

# Define LLM with bound tools
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")


# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
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
