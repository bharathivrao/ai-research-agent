AI Research Agent (Planner + RAG + Verifier)

A production-grade agentic AI system built with Python that performs structured research using planning, retrieval-augmented generation (RAG), verification, retries, metrics, and multi-tool execution, exposed via a FastAPI service.

This project demonstrates how to build reliable AI agents, not just chatbots.

⸻

🚀 What This Project Does

Given a research goal, the agent:
	1.	Plans the task into structured steps
	2.	Chooses the right tool per step
	•	rag → retrieve knowledge from documents
	•	code → perform calculations / analysis
	3.	Executes each step
	•	Uses embeddings + vector search for RAG
	•	Uses a safe Python execution tool for calculations
	4.	Verifies outputs
	•	Checks grounding against sources
	•	Flags hallucination risk
	5.	Retries automatically when confidence is low
	6.	Synthesizes a structured research report
	7.	Logs metrics (tokens, time, retries)
	8.	Exposes everything via an API


Architecture Overview

Goal
  ↓
Planner (JSON, tool-aware)
  ↓
Step Executor
  ├─ RAG Tool (OpenAI + Chroma)
  ├─ Code Tool (safe Python eval)
  ↓
Verifier / Evaluator
  ↓
Retry Loop (if needed)
  ↓
Final Research Report
  ↓
FastAPI Endpoint

Project Structure

ai-agent-week1/
│
├── planner.py        # Structured planner (chooses tools)
├── rag_chroma.py          # RAG layer (embeddings + Chroma)
├── tools.py               # Secondary tools (Python execution)
├── research_agent.py      # Orchestrator (plan → execute → verify → retry → report)
├── api_main.py            # FastAPI service
│
├── data/                  # Knowledge base (.txt files)
│   ├── ai_engineer_role.txt
│   ├── salary_samples.txt
│
├── chroma_db/             # Local vector DB (auto-generated)
├── .env                   # API keys (not committed)
├── requirements.txt
└── README.md

Tech Stack
	•	Python 3.12
	•	OpenAI API (LLMs + embeddings)
	•	ChromaDB (vector store)
	•	Pydantic v2 (schema enforcement)
	•	FastAPI (service layer)
	•	Uvicorn (ASGI server)

Setup Instructions

1️⃣ Clone & Create Virtual Environment

git clone https://github.com/your-username/ai-research-agent.git
cd ai-research-agent

python -m venv .venv
source .venv/bin/activate  # macOS/Linux

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Set Environment Variables

Create a .env file:
OPENAI_API_KEY=sk-xxxxxxxx

📚 Add Knowledge Base Documents

Add .txt files to the data/ directory.

Example:

AI Engineer salary samples:
- CompanyA: 185000
- CompanyB: 210000
- CompanyC: 195000

These files are embedded and indexed automatically.

▶️ Run the Agent (CLI)

python research_agent.py

Example goal:

goal = "Research AI Engineer job responsibilities and compute salary statistics."

🌐 Run as an API

uvicorn api_main:app --reload

Open Swagger UI:

👉 http://127.0.0.1:8000/docs

Example Request:

{
  "goal": "Research AI Engineer skills and compute average salary",
  "max_retries_per_step": 1
}

Example Response
	•	Structured plan
	•	Step-by-step execution (with tool used)
	•	Evaluations & retries
	•	Final research report
	•	Token & timing metrics

⸻

📊 Metrics Collected
	•	Tokens per step (input / output / total)
	•	Execution time per step
	•	Total pipeline time
	•	Retry counts
	•	Tool usage per step

This makes the agent observable, debuggable, and cost-aware.

🧪 Tooling

🔍 RAG Tool
	•	OpenAI embeddings (text-embedding-3-small)
	•	Chroma vector search
	•	Source attribution

🧮 Code Tool
	•	LLM-generated Python expressions
	•	Safe evaluation (restricted environment)
	•	Used only for numeric/logical steps

✅ Why This Project Matters

This project demonstrates real AI engineering, including:
	•	Agentic workflows
	•	Tool orchestration
	•	Reliability & verification
	•	Self-correction loops
	•	Production-ready APIs
	•	Observability & metrics


⸻

🛣️ Possible Extensions
	•	Web search tool
	•	SQL / data warehouse tool
	•	Long-term memory
	•	Async execution
	•	Streaming responses
	•	Cost budgeting
	•	Auth & rate limiting
