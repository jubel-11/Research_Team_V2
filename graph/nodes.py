"""
Node functions for Research Team v2.

Nodes:
  planner_node      → breaks topic into research plan
  dispatcher_node   → fans out to 3 parallel researchers (Send API)
  researcher_node   → one researcher instance (runs 3x in parallel)
  critic_node       → reflects on combined findings, scores quality
  writer_node       → writes final report from findings + critique
  human_approval_node → HITL gate before publishing
  publisher_node    → handles approve/edit/reject decision
"""

import os
import re
import json
import time
from typing import Annotated
from operator import add

from langgraph.constants import Send
from langgraph.types     import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from graph.state import ResearchState

load_dotenv()

# ─────────────────────────────────────────────
# LLM — Gemini 2.5 Flash Lite
# 1,000 requests/day free — plenty for this project
# ─────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)

# Research angles for parallel execution
RESEARCH_ANGLES = [
    "Technical & Scientific",
    "Economic & Industry",
    "Social & Ethical",
]


def _call_llm(prompt: str, label: str = "") -> str:
    """Call Gemini with rate limit handling."""
    for attempt in range(3):
        try:
            if label:
                print(f"    📡 [{label}]...")
            response = llm.invoke(prompt)
            time.sleep(3)  # rate limit buffer
            return response.content.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"    ⚠️  Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"Error: {str(e)}"
    return "Error: max retries exceeded"


# ─────────────────────────────────────────────
# NODE 1 — Planner
# ─────────────────────────────────────────────

def planner_node(state: ResearchState) -> dict:
    """
    Planner: breaks the topic into a structured research plan.
    Identifies key questions for each research angle.
    """
    topic = state["topic"]

    print(f"\n{'─'*55}")
    print(f"  📋 PLANNER NODE")
    print(f"{'─'*55}")
    print(f"  Topic: {topic}")

    prompt = f"""You are a research planning expert.

Create a structured research plan for: "{topic}"

For each of these 3 angles, define 2-3 specific research questions:
1. Technical & Scientific
2. Economic & Industry  
3. Social & Ethical

Reply with ONLY this JSON:
{{
  "topic": "{topic}",
  "angles": {{
    "Technical & Scientific": ["question 1", "question 2"],
    "Economic & Industry": ["question 1", "question 2"],
    "Social & Ethical": ["question 1", "question 2"]
  }},
  "key_objective": "one sentence research objective"
}}"""

    raw = _call_llm(prompt, "Planner")
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        plan  = json.loads(clean)
    except Exception:
        plan = {
            "topic":         topic,
            "angles":        {a: [f"Research {a} aspects of {topic}"] for a in RESEARCH_ANGLES},
            "key_objective": f"Understand the impact of {topic}",
        }

    print(f"  ✅ Plan created: {plan.get('key_objective', '')[:60]}")
    for angle in RESEARCH_ANGLES:
        questions = plan.get("angles", {}).get(angle, [])
        print(f"     [{angle}]: {len(questions)} questions")

    return {
        "research_plan":     plan,
        "research_findings": [],
        "iteration":         1,
    }


def dispatcher_routing(state: ResearchState):
    """
    Routing function that returns Send objects for parallel researcher execution.
    This is called by conditional edges after the planner node.
    """
    topic = state["topic"]
    plan  = state.get("research_plan", {})

    print(f"\n{'─'*55}")
    print(f"  📡 DISPATCHER — Launching 3 parallel researchers")
    print(f"{'─'*55}")

    sends = []
    for angle in RESEARCH_ANGLES:
        questions = plan.get("angles", {}).get(angle, [f"Research {angle} aspects"])
        print(f"  → [{angle}]")
        sends.append(Send("researcher", {
            "topic":    topic,
            "angle":    angle,
            "questions": questions,
        }))

    return sends


# ─────────────────────────────────────────────
# NODE 2 — Dispatcher (Map / Fan-Out)
# ─────────────────────────────────────────────

def dispatcher_node(state: ResearchState):
    """
    Dispatcher: fans out to 3 parallel researchers using Send API.
    Each researcher gets the topic + their specific angle + questions.

    Returns Send() objects — LangGraph runs them simultaneously.
    This is the MAP step of map-reduce.
    """
    topic = state["topic"]
    plan  = state.get("research_plan", {})

    print(f"\n{'─'*55}")
    print(f"  📡 DISPATCHER — Launching 3 parallel researchers")
    print(f"{'─'*55}")

    sends = []
    for angle in RESEARCH_ANGLES:
        questions = plan.get("angles", {}).get(angle, [f"Research {angle} aspects"])
        print(f"  → [{angle}]")
        sends.append(Send("researcher", {
            "topic":    topic,
            "angle":    angle,
            "questions": questions,
        }))

    return sends


# ─────────────────────────────────────────────
# NODE 3 — Researcher (runs 3x in parallel)
# ─────────────────────────────────────────────

def researcher_node(state: dict) -> dict:
    """
    Researcher: researches one specific angle of the topic.
    Runs 3 times simultaneously — one per angle.

    Returns findings that get APPENDED to research_findings list
    in the parent state (Annotated[list, add] handles the merge).
    """
    topic     = state["topic"]
    angle     = state["angle"]
    questions = state.get("questions", [])

    print(f"    🔬 Researcher [{angle}] starting...")

    questions_text = "\n".join(f"- {q}" for q in questions)
    prompt = f"""You are a specialist researcher focused on: {angle}

Research topic: "{topic}"

Answer these specific questions from your specialist perspective:
{questions_text}

Provide detailed, factual findings with:
- Specific data points or statistics where possible
- Real examples or case studies
- Expert insights from the {angle} perspective
- 3-4 key findings formatted as numbered points

Be thorough and specific — not vague generalities."""

    findings = _call_llm(prompt, f"Researcher [{angle}]")

    formatted = f"[{angle.upper()} FINDINGS]\n{findings}"
    print(f"    ✅ Researcher [{angle}] complete ({len(findings)} chars)")

    return {
        "research_findings": [formatted],
    }


# ─────────────────────────────────────────────
# NODE 4 — Critic (Reflection)
# ─────────────────────────────────────────────

def critic_node(state: ResearchState) -> dict:
    """
    Critic: reviews all research findings and scores quality.
    Identifies gaps, weaknesses, and what needs improvement.

    This is the REFLECTION pattern from Week 1 applied to
    multi-agent research — the critic improves output quality
    before the writer produces the final report.
    """
    topic    = state["topic"]
    findings = state["research_findings"]

    print(f"\n{'─'*55}")
    print(f"  🔍 CRITIC NODE (Reflection)")
    print(f"{'─'*55}")
    print(f"  Reviewing {len(findings)} research streams...")

    all_findings = "\n\n".join(findings)

    prompt = f"""You are a critical research evaluator.

Topic: "{topic}"

Review these research findings from 3 specialist researchers:
{all_findings[:2000]}

Evaluate:
1. Coverage — are all important aspects covered?
2. Depth — are findings specific enough with data?
3. Balance — are all 3 perspectives well represented?
4. Gaps — what important aspects are missing?

Reply with ONLY this JSON:
{{
  "score": 7,
  "needs_revision": false,
  "strengths": ["strength 1", "strength 2"],
  "gaps": ["gap 1", "gap 2"],
  "recommendations": ["improve X", "add Y"],
  "summary": "one paragraph critique"
}}

Score 8+ = good to write. Below 8 = note gaps but proceed."""

    raw = _call_llm(prompt, "Critic")
    try:
        clean  = re.sub(r"```(?:json)?|```", "", raw).strip()
        data   = json.loads(clean)
        score  = int(data.get("score", 7))
        needs  = data.get("needs_revision", False)
        summary = data.get("summary", "")
    except Exception:
        score   = 7
        needs   = False
        summary = raw[:300]
        data    = {"score": score, "needs_revision": needs, "summary": summary}

    print(f"  📊 Critique score: {score}/10")
    print(f"  Needs revision  : {needs}")
    print(f"  Summary: {summary[:80]}...")

    return {
        "critique":       json.dumps(data),
        "critique_score": score,
        "needs_revision": needs,
    }


# ─────────────────────────────────────────────
# NODE 5 — Writer
# ─────────────────────────────────────────────

def writer_node(state: ResearchState) -> dict:
    """
    Writer: synthesizes all findings + critique into a polished report.
    Uses the critic's feedback to address gaps before writing.
    """
    topic    = state["topic"]
    findings = state["research_findings"]
    critique = state.get("critique", "{}")
    score    = state.get("critique_score", 7)

    print(f"\n{'─'*55}")
    print(f"  ✍️  WRITER NODE")
    print(f"{'─'*55}")
    print(f"  Writing from {len(findings)} research streams (critic score: {score}/10)")

    try:
        critique_data = json.loads(critique)
        gaps          = critique_data.get("gaps", [])
        recommendations = critique_data.get("recommendations", [])
    except Exception:
        gaps            = []
        recommendations = []

    all_findings   = "\n\n".join(findings)
    gaps_text      = "\n".join(f"- {g}" for g in gaps) if gaps else "None identified"
    recs_text      = "\n".join(f"- {r}" for r in recommendations) if recommendations else "None"

    prompt = f"""You are an expert research writer.

Write a comprehensive research report on: "{topic}"

Use these research findings:
{all_findings[:2500]}

The critic identified these gaps to address:
{gaps_text}

Recommendations to incorporate:
{recs_text}

Write a polished report (500-600 words) with:
1. Executive Summary (2-3 sentences)
2. Technical & Scientific Analysis
3. Economic & Industry Perspective
4. Social & Ethical Considerations
5. Key Findings (5 bullet points)
6. Conclusion

Use professional language. Reference specific facts from the research."""

    draft = _call_llm(prompt, "Writer")

    print(f"  ✅ Draft complete ({len(draft)} chars)")
    return {"draft_report": draft}


# ─────────────────────────────────────────────
# NODE 6 — Human Approval (HITL)
# ─────────────────────────────────────────────

def human_approval_node(state: ResearchState) -> Command:
    """
    Human Approval Gate: pauses graph, saves checkpoint, waits for human.

    Uses interrupt() to freeze execution — MemorySaver saves full state.
    Human can approve, edit, or reject before publishing.
    Resumes with Command(resume=...) after human input.
    """
    draft     = state["draft_report"]
    topic     = state["topic"]
    score     = state.get("critique_score", 7)
    thread_id = state["thread_id"]

    print(f"\n{'─'*55}")
    print(f"  🚦 HUMAN APPROVAL GATE")
    print(f"  [Graph PAUSED — State SAVED to checkpoint: {thread_id}]")
    print(f"{'─'*55}")

    human_input = interrupt({
        "message":   "Research report ready for human review",
        "topic":     topic,
        "draft":     draft,
        "score":     score,
        "thread_id": thread_id,
        "options":   ["approve", "edit", "reject"],
    })

    decision = human_input.get("decision", "approve")
    feedback = human_input.get("feedback", "")

    print(f"\n  Human decision: {decision.upper()}")
    if feedback:
        print(f"  Feedback: {feedback[:80]}")

    return Command(
        update={
            "human_decision": decision,
            "human_feedback": feedback,
        },
        goto="publisher"
    )


# ─────────────────────────────────────────────
# NODE 7 — Publisher
# ─────────────────────────────────────────────

def publisher_node(state: ResearchState) -> dict:
    """
    Publisher: handles the human's decision.
    approve → publish as-is
    edit    → rewrite with feedback
    reject  → do not publish
    """
    decision = state.get("human_decision", "approve")
    feedback = state.get("human_feedback", "")
    draft    = state["draft_report"]
    topic    = state["topic"]

    print(f"\n{'─'*55}")
    print(f"  📢 PUBLISHER NODE — {decision.upper()}")
    print(f"{'─'*55}")

    if decision == "reject":
        print(f"  ❌ Report rejected — not publishing")
        return {"final_report": None, "published": False}

    elif decision == "edit":
        print(f"  ✏️  Applying human feedback...")
        prompt = (
            f"Rewrite this research report on '{topic}' "
            f"applying this feedback:\n\n"
            f"Feedback: {feedback}\n\n"
            f"Original report:\n{draft}\n\n"
            f"Incorporate all feedback while keeping the structure intact:"
        )
        final = _call_llm(prompt, "Publisher [rewrite]")
        print(f"  ✅ Report rewritten ({len(final)} chars)")
        return {"final_report": final, "published": True}

    else:  # approve
        print(f"  ✅ Report approved — publishing")
        return {"final_report": draft, "published": True}
