# api_main.py

from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from research_agent import run_research_pipeline


app = FastAPI(
    title="AI Research Agent",
    version="0.1.0",
    description=(
        "Decision + Retrieval Research Agent "
        "(planner + RAG + evaluator + retry + metrics)."
    ),
)


# ----------- Request / Response Models -----------

class ResearchRequest(BaseModel):
    goal: str
    max_retries_per_step: int = 1


class ResearchResponse(BaseModel):
    goal: str
    plan: Dict[str, Any]
    steps: List[Dict[str, Any]]
    report: Dict[str, Any]
    metrics: Dict[str, Any]


# ----------- Endpoint -----------

@app.post("/research", response_model=ResearchResponse)
def run_research(req: ResearchRequest) -> ResearchResponse:
    """
    Run the full research pipeline for a given goal.
    Returns:
    - goal
    - plan (structured plan from the planner)
    - steps (RAG answers + evaluations + retries + per-step metrics)
    - report (structured research report)
    - metrics (pipeline totals: tokens, time, etc.)
    """
    result = run_research_pipeline(
        goal=req.goal,
        max_retries_per_step=req.max_retries_per_step,
    )

    return ResearchResponse(
        goal=result["goal"],
        plan=result["plan"],
        steps=result["steps"],
        report=result["report"],
        metrics=result["metrics"],
    )