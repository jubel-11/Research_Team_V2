"""
State definitions for Research Team v2.
All nodes read from and write to ResearchState.
"""

from typing import TypedDict, Optional, Annotated
from operator import add


class ResearchState(TypedDict):
    """
    Shared state flowing through all nodes.

    research_findings uses Annotated[list, add] for map-reduce:
    3 parallel researchers each append their findings — no conflicts.
    """
    # Input
    topic:              str

    # Planner output
    research_plan:      Optional[dict]     # sections + angles to research

    # Parallel researcher outputs (map-reduce)
    research_findings:  Annotated[list, add]

    # Critic output
    critique:           Optional[str]
    critique_score:     Optional[int]       # 1-10
    needs_revision:     bool

    # Writer output
    draft_report:       Optional[str]

    # Human approval
    human_decision:     Optional[str]       # approve / edit / reject
    human_feedback:     Optional[str]

    # Final
    final_report:       Optional[str]
    published:          bool

    # Metadata
    thread_id:          str
    iteration:          int
