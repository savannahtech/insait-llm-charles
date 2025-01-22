from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from agent import llm, retrieve_return_policy
from prompts import generator_rag_prompt, grader_prompt, rewriter_prompt, system_message
from tools import check_order_status, save_user_info

tools = [check_order_status, save_user_info, retrieve_return_policy]
tool_node = ToolNode(tools)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Conversation summerizer Node
def summarize_conversation(state: AgentState):
    summary = ...


def get_query(messages: Sequence[BaseMessage]):
    query = "ask me to rephrase"
    for message in reversed(messages):
        if message.type == "human":
            query = message.content
            break
    return query


def get_retrieved_documents(messages: Sequence[BaseMessage]):
    recent_retriever_messages = []
    for message in reversed(messages):
        if message.type == "tool" and message.name == "return_policy":
            recent_retriever_messages.append(message)
        else:
            break

    retriever_messages = recent_retriever_messages[::-1]
    return "\n\n".join(doc.content for doc in retriever_messages)


def get_conversation(messages: Sequence[BaseMessage]):
    return [
        message
        for message in messages
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]


def route_to_tools(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


def grade_documents(state: AgentState):
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm_with_structured_op = llm.with_structured_output(grade)

    chain = grader_prompt | llm_with_structured_op

    query = get_query(state["messages"])
    docs = get_retrieved_documents(state["messages"])

    scored_result = chain.invoke({"question": query, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        return "generator"
    else:
        return "rewriter"


def rewrite(state: AgentState):
    chain = rewriter_prompt | llm

    query = get_query(state["messages"])
    conversation = get_conversation(state["messages"])

    print("Rewriting query...")

    response = chain.invoke({"question": query, "conversation": conversation})
    return {"messages": [response]}


def generate(state: AgentState):
    query = get_query(state["messages"])
    context = get_retrieved_documents(state["messages"])
    conversation = get_conversation(state["messages"])

    chain = generator_rag_prompt | llm

    response = chain.invoke({"user_input": query, "context": context, "conversation": conversation})
    return {"messages": [response]}


# Agent (Chatbot) node
def agent(state: AgentState):
    model = llm.bind_tools(tools)
    response = model.invoke([system_message] + state["messages"])
    return {"messages": [response]}


def build_graph():
    # Graph
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("generator", generate)
    workflow.add_node("rewriter", rewrite)
    # workflow.add_node(summarize_conversation)

    # Define edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_to_tools, ["tools", END])
    workflow.add_conditional_edges("tools", grade_documents, ["generator", "rewriter"])

    # workflow.add_edge("tools", "agent")
    workflow.add_edge("rewriter", "agent")
    workflow.add_edge("generator", END)

    return workflow
