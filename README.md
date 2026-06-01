# 🤖 Research Team v2

A production-ready multi-agent research pipeline using LangGraph into a single cohesive system.

## 🧠 Architecture

```
START
  ↓
[Planner]          → breaks topic into structured research plan
  ↓
[Dispatcher]       → fans out to 3 researchers simultaneously (Send API)
  ↓  ↓  ↓
[Researcher x3]    → Technical | Economic | Social (parallel execution)
  ↓
[Critic]           → reflects on findings, scores quality, identifies gaps
  ↓
[Writer]           → synthesizes findings + critique into polished report
  ↓
[Human Approval]   ← INTERRUPT: graph paused, checkpoint saved
  ↓
[Publisher]        → approve / edit / reject
  ↓
END
```

## 🔬 LangGraph Concepts Demonstrated

| Concept | Implementation |
|---|---|
| **Multi-node graph** | 7 nodes in structured pipeline |
| **Parallel execution** | Send API fans out to 3 researchers simultaneously |
| **Map-reduce** | `Annotated[list, add]` merges 3 findings streams |
| **Subgraph modularity** | Each researcher is an independent processing unit |
| **Reflection/Critique** | Critic node scores and identifies gaps before writing |
| **Human-in-the-loop** | `interrupt()` pauses graph at approval gate |
| **Checkpoint persistence** | `MemorySaver` saves state at every node |
| **Resume from checkpoint** | `Command(resume=...)` continues after human input |
| **Time-travel debugging** | `get_state_history()` shows all checkpoints |
| **LangSmith tracing** | Full observability — latency, tokens, node traces |
| **Streaming** | `stream_mode="updates"` shows real-time progress |

## 📁 Project Structure

```
research-team-v2/
├── .env                    # API keys (not committed)
├── main.py                 # Entry point + HITL runner
├── requirements.txt
├── graph/
│   ├── state.py            # ResearchState TypedDict
│   ├── nodes.py            # All 7 node functions
│   └── builder.py          # StateGraph construction
└── reports/                # Generated reports saved here
```

## 🚀 Setup & Run

```bash
# 1. Virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. Environment variables
# Create .env with:
GEMINI_API_KEY=your-gemini-key
LANGCHAIN_API_KEY=your-langsmith-key   # optional but recommended

# 4. Run
python main.py
```

## 💡 Sample Output

```
🤖 RESEARCH TEAM v2
  Topic: The impact of AI on the future of work

📋 PLANNER NODE — creating research plan...
📡 DISPATCHER — launching 3 parallel researchers
  → [Technical & Scientific]
  → [Economic & Industry]
  → [Social & Ethical]
    ✅ Researcher [Economic] complete (2.1s)
    ✅ Researcher [Social] complete (2.3s)
    ✅ Researcher [Technical] complete (2.7s)
🔍 CRITIC NODE — scoring: 8/10
✍️  WRITER NODE — synthesizing...

🚦 HUMAN APPROVAL GATE — [Graph PAUSED]
  Your decision: approve

✅ PUBLISHED — report saved to reports/
🔗 View traces: smith.langchain.com
```

## 🔧 Tech Stack

- **LangGraph** — graph orchestration
- **Gemini 2.5 Flash Lite** — LLM
- **LangSmith** — tracing and observability
- **Python** — pure Python, no additional frameworks

## 📚 References

- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [LangSmith Docs](https://docs.smith.langchain.com)
- [Reflexion Paper — Shinn et al. 2023](https://arxiv.org/abs/2303.11366)

## 👨‍💻 Author

**Jubelin Joji**
