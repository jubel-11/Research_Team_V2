"""
Research Team v2 — Main Entry Point
Week 3 Project: Planner → 3 Parallel Researchers → Critic → Writer → HITL → Publisher
"""

import os
import uuid
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─── LangSmith Tracing Setup ──────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = "research-team-v2"
os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"

TRACING = bool(os.getenv("LANGCHAIN_API_KEY"))
if TRACING:
    print("✅ LangSmith tracing → project: 'research-team-v2'")
else:
    print("⚠️  LangSmith disabled (add LANGCHAIN_API_KEY to .env)")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types             import Command
from graph                       import build_graph, ResearchState


DEFAULT_TOPIC = "The impact of artificial intelligence on the future of work"


def save_report(report: str, topic: str):
    """Save the final report to a markdown file."""
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug      = topic.lower().replace(" ", "_")[:30]
    filename  = f"reports/{slug}_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Research Report: {topic}\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write("---\n\n")
        f.write(report)

    print(f"\n  💾 Report saved to '{filename}'")
    return filename


def show_checkpoint_history(graph, config: dict):
    """Show all saved checkpoints for time-travel debugging."""
    print(f"\n{'─'*55}")
    print(f"  🕐 CHECKPOINT HISTORY")
    print(f"{'─'*55}")

    history = list(graph.get_state_history(config))
    print(f"  Total checkpoints: {len(history)}")

    for i, snap in enumerate(reversed(history)):
        step = len(history) - i
        vals = snap.values
        print(f"\n  Checkpoint {step} → next: {snap.next}")
        print(f"    Plan ready   : {'✅' if vals.get('research_plan') else '❌'}")
        print(f"    Findings     : {len(vals.get('research_findings', []))} streams")
        print(f"    Critic score : {vals.get('critique_score', 'pending')}")
        print(f"    Draft ready  : {'✅' if vals.get('draft_report') else '❌'}")
        print(f"    Decision     : {vals.get('human_decision', 'pending')}")
        print(f"    Published    : {vals.get('published', False)}")


def run_research_team(topic: str):
    """
    Run the full Research Team v2 pipeline.

    Phase 1: Planner → Dispatcher → 3 Parallel Researchers → Critic → Writer
    Phase 2: INTERRUPT — show draft to human, get decision
    Phase 3: Resume → Publisher
    Phase 4: Save report + show checkpoint history
    """
    checkpointer = MemorySaver()
    graph        = build_graph(checkpointer)

    thread_id = str(uuid.uuid4())[:8]
    config    = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'='*55}")
    print(f"  🤖 RESEARCH TEAM v2")
    print(f"  Topic    : {topic}")
    print(f"  Thread ID: {thread_id}")
    print(f"  Nodes    : Planner → [3x Researcher] → Critic → Writer → Human → Publisher")
    print(f"{'='*55}")

    initial_state: ResearchState = {
        "topic":              topic,
        "research_plan":      None,
        "research_findings":  [],
        "critique":           None,
        "critique_score":     None,
        "needs_revision":     False,
        "draft_report":       None,
        "human_decision":     None,
        "human_feedback":     None,
        "final_report":       None,
        "published":          False,
        "thread_id":          thread_id,
        "iteration":          1,
    }

    # ── PHASE 1: Run until human approval gate ──
    print(f"\n  📍 PHASE 1: Running pipeline until approval gate...")
    print(f"  [Planner → Researchers (parallel) → Critic → Writer]\n")

    start_time     = time.time()
    interrupted_state = None

    for event in graph.stream(initial_state, config, stream_mode="values"):
        interrupted_state = event

    phase1_time = time.time() - start_time
    print(f"\n  ⏸️  Graph paused at approval gate ({phase1_time:.1f}s elapsed)")

    # ── PHASE 2: Show draft to human ──
    draft = interrupted_state.get("draft_report", "") if interrupted_state else ""
    score = interrupted_state.get("critique_score", "?") if interrupted_state else "?"

    print(f"\n{'='*55}")
    print(f"  📄 DRAFT REPORT FOR REVIEW (Critic score: {score}/10)")
    print(f"{'='*55}")
    print(draft[:1000] + ("..." if len(draft) > 1000 else ""))

    print(f"\n{'─'*55}")
    print(f"  🧑 HUMAN APPROVAL REQUIRED")
    print(f"{'─'*55}")
    print(f"  approve → publish as-is")
    print(f"  edit    → provide feedback and rewrite")
    print(f"  reject  → do not publish")
    print()

    decision = input("  Your decision (approve/edit/reject): ").strip().lower()
    if decision not in ["approve", "edit", "reject"]:
        decision = "approve"

    feedback = ""
    if decision == "edit":
        feedback = input("  Your feedback: ").strip()
    elif decision == "reject":
        feedback = input("  Reason for rejection: ").strip()

    # ── PHASE 3: Resume graph ──
    print(f"\n  📍 PHASE 3: Resuming from checkpoint '{thread_id}'...")

    final_state = None
    for event in graph.stream(
        Command(resume={"decision": decision, "feedback": feedback}),
        config,
        stream_mode="values"
    ):
        final_state = event

    total_time = time.time() - start_time

    # ── PHASE 4: Results ──
    print(f"\n{'='*55}")
    print(f"  📊 FINAL RESULT")
    print(f"{'='*55}")
    print(f"  Total time : {total_time:.1f}s")
    print(f"  Thread ID  : {thread_id}")
    print(f"  Decision   : {decision.upper()}")

    if final_state:
        published = final_state.get("published", False)
        final     = final_state.get("final_report")

        if published and final:
            print(f"  Status     : ✅ PUBLISHED")
            print(f"\n  Final Report Preview:")
            print(f"  {'─'*50}")
            print(final[:600] + ("..." if len(final) > 600 else ""))
            save_report(final, topic)
        else:
            print(f"  Status     : ❌ NOT PUBLISHED")

    if TRACING:
        print(f"\n  🔗 View traces at: https://smith.langchain.com")
        print(f"  Project: research-team-v2")

    show_checkpoint_history(graph, config)


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in .env")
        exit(1)

    print("\n🤖 Research Team v2 — Week 3 Project")
    print("=" * 55)
    print("1 → Demo topic (AI impact on future of work)")
    print("2 → Enter your own topic")

    choice = input("\nYour choice (1 or 2): ").strip()

    if choice == "2":
        topic = input("Enter research topic: ").strip()
        if not topic:
            topic = DEFAULT_TOPIC
    else:
        topic = DEFAULT_TOPIC

    run_research_team(topic)
