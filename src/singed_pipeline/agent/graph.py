from langgraph.graph import END,START, StateGraph

from singed_pipeline.agent.nodes import (
    answer_node, 
    grade_evidence_node,
    refuse_node,
    retrieve_node,
    rewrite_query_node,
    route_after_grading,
)

from singed_pipeline.agent.state import AgentState

builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_evidence_node)
builder.add_node("rewrite", rewrite_query_node)
builder.add_node("answer", answer_node)
builder.add_node("refuse", refuse_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")

builder.add_conditional_edges(
    "grade",
    route_after_grading,
    {
        "answer": "answer",
        "rewrite": "rewrite",
        "refuse": "refuse",
    },
)

builder.add_edge("rewrite", "retrieve")
builder.add_edge("answer", END)
builder.add_edge("refuse", END)

rag_graph = builder.compile()
