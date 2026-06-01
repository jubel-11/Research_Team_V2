"""
Graph builder for Research Team v2.

Full graph flow:
  START
    → planner
    → dispatcher → [researcher x3 in parallel]  ← map / fan-out
    → critic                                      ← reduce / fan-in
    → writer
    → human_approval  [INTERRUPT — checkpoint saved]
    → publisher
    → END
"""

import os
from langgraph.graph             import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import ResearchState
from graph.nodes import (
    planner_node,
    dispatcher_node,
    researcher_node,
    critic_node,
    writer_node,
    human_approval_node,
    publisher_node,
)


def build_graph(checkpointer: MemorySaver):
    """
    Build and compile the Research Team v2 StateGraph.

    Combines all Week 3 patterns:
      ✅ Multi-node graph     : 7 nodes
      ✅ Parallel execution   : dispatcher → 3 researchers via Send API
      ✅ Map-reduce           : Annotated[list, add] merges findings
      ✅ Reflection           : critic reviews before writer
      ✅ Human-in-the-loop    : interrupt() at approval gate
      ✅ Checkpoint persistence: MemorySaver saves at every node
      ✅ LangSmith tracing    : automatic via env vars
    """
    from graph.nodes import dispatcher_routing
    
    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("planner",        planner_node)
    graph.add_node("researcher",     researcher_node)
    graph.add_node("critic",         critic_node)
    graph.add_node("writer",         writer_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("publisher",      publisher_node)

    # Fixed edges — sequential flow
    graph.add_edge(START,          "planner")
    
    # Use conditional edges from planner to handle Send objects
    graph.add_conditional_edges(
        "planner",
        dispatcher_routing  # This function returns Send objects for parallel execution
    )
    
    graph.add_edge("researcher",   "critic")     # fan-in after parallel
    graph.add_edge("critic",       "writer")
    graph.add_edge("writer",       "human_approval")
    # human_approval uses Command(goto="publisher")
    graph.add_edge("publisher",    END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])
