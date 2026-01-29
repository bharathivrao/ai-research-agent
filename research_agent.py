# research_agent.py
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import time
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from tools import run_python_tool
from planner import make_plan, Plan
from rag_chroma import answer_with_rag, ingest_folder, get_docs_by_ids

load_dotenv()  # load OPENAI_API_KEY from .env
client = OpenAI()


# ---------- Structured Research Report models ----------

class ResearchSection(BaseModel):
    title: str = Field(..., description="Section title")
    summary: str = Field(..., description="2–4 sentence summary of this section")
    bullet_points: List[str] = Field(
        default_factory=list,
        description="Key bullet points for this section",
    )


class ResearchReport(BaseModel):
    topic: str = Field(..., description="The research topic / user goal")
    overview: str = Field(..., description="High-level overview of findings")
    key_findings: List[str] = Field(
        default_factory=list,
        description="3–7 key findings as short sentences",
    )
    sections: List[ResearchSection] = Field(
        default_factory=list,
        description="Major sections of the report",
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="Limitations, uncertainties, or missing information",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of source document ids used (from the RAG layer)",
    )


# ---------- Evaluation models ----------

class StepEvaluation(BaseModel):
    step_index: int
    supported_by_sources: bool = Field(
        ...,
        description="True if the answer is well-supported by the provided source documents.",
    )
    hallucination_risk: str = Field(
        ...,
        description="One of: 'low', 'medium', 'high'.",
    )
    missing_or_ambiguous_info: bool = Field(
        ...,
        description="True if important info is missing or unclear in the sources.",
    )
    overall_quality: str = Field(
        ...,
        description="Short textual rating like 'good', 'ok', 'poor'.",
    )
    comments: str = Field(
        ...,
        description="Short explanation of the evaluation, including any concerns.",
    )


# ---------- Core Agent Logic ----------

def run_research_pipeline(goal: str, max_retries_per_step: int = 1) -> Dict[str, Any]:
    pipeline_start = time.perf_counter()
    """
    1) Plan the research task.
    2) Run RAG for each step description (with retries on bad evals).
    3) Evaluate each step's answer vs its sources.
    4) Synthesize a structured research report.
    """
    # 0) Make sure data is ingested once at start of app
    ingest_folder("data")

    # 1) Planning
    plan: Plan = make_plan(goal)

    step_results: List[Dict[str, Any]] = []

    # 2) Execute each step with RAG + evaluation + retry loop
    memory = {"facts": []}

    for idx, step in enumerate(plan.steps):
        step_result = execute_step_with_retry(
            step_index=idx,
            description=step.description,
            depends_on=step.depends_on,
            tool=step.tool,
            memory=memory,
            max_retries=max_retries_per_step,
        )
    step_results.append(step_result)

    # Save step outputs in memory for later steps (esp. for code steps)
    memory["facts"].append({
        "step_index": idx,
        "tool": step.tool,
        "description": step.description,
        "answer": step_result["answer"],
        "sources": step_result.get("sources", []),
    })

    # Aggregate metrics across steps
    total_input_tokens = 0
    total_output_tokens = 0
    total_step_time = 0.0

    for s in step_results:
        usage = s.get("token_usage", {})
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)
        total_step_time += s.get("elapsed_seconds", 0.0)

    # 3) Synthesize research report (aware of evaluations)
    report = synthesize_research_report(goal, plan, step_results)

    pipeline_elapsed = time.perf_counter() - pipeline_start

    # Basic logging to stdout
    print("\n=== PIPELINE METRICS ===")
    print(
        f"Total input tokens (RAG + eval, steps only): {total_input_tokens}, "
        f"output tokens: {total_output_tokens}, "
        f"total: {total_input_tokens + total_output_tokens}"
    )
    print(
        f"Total step time: {total_step_time:.2f}s, "
        f"pipeline wall time (incl. ingest+plan+report): {pipeline_elapsed:.2f}s"
    )

    return {
        "goal": goal,
        "plan": plan.model_dump(),
        "steps": step_results,
        "report": report.model_dump(),
        "metrics": {
            "total_input_tokens_steps": total_input_tokens,
            "total_output_tokens_steps": total_output_tokens,
            "total_tokens_steps": total_input_tokens + total_output_tokens,
            "total_step_time_seconds": total_step_time,
            "pipeline_time_seconds": pipeline_elapsed,
        },
    }


def execute_step_with_retry(
    step_index: int,
    description: str,
    depends_on: List[int],
    tool: str,
    memory: Dict[str, Any],
    max_retries: int = 1,
) -> Dict[str, Any]:
    """
    Execute one step with RAG, evaluate it, and retry if the evaluation is bad.
    Returns a dict with final answer, sources, evaluation, num_attempts,
    token_usage, elapsed_seconds, and all attempts (for debugging).
    """
    
    attempts: List[Dict[str, Any]] = []
    k = 3  # start with top-3 docs

    total_input_tokens = 0
    total_output_tokens = 0

    start_time = time.perf_counter()

    if tool == "rag":
        # existing RAG + eval + retry loop
        for attempt in range(max_retries + 1):
            answer, sources, usage_rag = answer_with_rag(description, k=k)
            source_docs = get_docs_by_ids(sources)

            evaluation, usage_eval = evaluate_step_answer(
                step_index=step_index,
                question=description,
                answer=answer,
                source_ids=sources,
                source_docs=source_docs,
            )

            total_input_tokens += usage_rag["input_tokens"] + usage_eval["input_tokens"]
            total_output_tokens += usage_rag["output_tokens"] + usage_eval["output_tokens"]

            attempt_record = {
                "attempt_index": attempt,
                "tool": tool,
                "answer": answer,
                "sources": sources,
                "evaluation": evaluation.model_dump(),
                "usage_rag": usage_rag,
                "usage_eval": usage_eval,
            }
            attempts.append(attempt_record)

            good_enough = (
                evaluation.supported_by_sources
                and evaluation.hallucination_risk in ("low", "medium")
                and not evaluation.missing_or_ambiguous_info
                and evaluation.overall_quality in ("good", "ok")
            )

            if good_enough:
                break

            k = min(k + 2, 8)

    elif tool == "code":
        # Single-shot code tool (no RAG, no cross-doc evaluation)
        memory_json = json.dumps(memory, ensure_ascii=False)
        tool_result = run_python_tool(
            instruction=(
                "You are computing based on prior extracted facts.\n\n"
                f"MEMORY_JSON:\n{memory_json}\n\n"
                f"TASK:\n{description}\n\n"
                "Important: Use any numeric values found in MEMORY_JSON if relevant."
            )
        )
        usage = tool_result["usage"]

        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]

        # For now, we create a trivial evaluation for 'code' steps
        evaluation = StepEvaluation(
            step_index=step_index,
            supported_by_sources=True,    # no external sources; tool is deterministic
            hallucination_risk="low",
            missing_or_ambiguous_info=False,
            overall_quality="good" if tool_result["status"] == "ok" else "poor",
            comments=(
                "Code tool executed successfully."
                if tool_result["status"] == "ok"
                else f"Code tool error: {tool_result['error']}"
            ),
        )

        attempt_record = {
            "attempt_index": 0,
            "tool": tool,
            "answer": str(tool_result["result"]),
            "sources": [],  # no RAG sources
            "evaluation": evaluation.model_dump(),
            "usage_tool": usage,
            "expression": tool_result["expression"],
        }
        attempts.append(attempt_record)

    else:
        # Unknown tool: fail fast
        evaluation = StepEvaluation(
            step_index=step_index,
            supported_by_sources=False,
            hallucination_risk="high",
            missing_or_ambiguous_info=True,
            overall_quality="poor",
            comments=f"Unknown tool '{tool}'.",
        )
        attempt_record = {
            "attempt_index": 0,
            "tool": tool,
            "answer": "",
            "sources": [],
            "evaluation": evaluation.model_dump(),
        }
        attempts.append(attempt_record)

    elapsed = time.perf_counter() - start_time
    final = attempts[-1]

    return {
        "step_index": step_index,
        "description": description,
        "depends_on": depends_on,
        "tool": tool,
        "answer": final["answer"],
        "sources": final.get("sources", []),
        "evaluation": final["evaluation"],
        "num_attempts": len(attempts),
        "attempts": attempts,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
        "elapsed_seconds": elapsed,
    }


def evaluate_step_answer(
    step_index: int,
    question: str,
    answer: str,
    source_ids: List[str],
    source_docs: List[str],
) -> StepEvaluation:
    """
    Ask the LLM to act as a verifier:
    - Compare the step answer against the source documents.
    - Judge support, hallucination risk, and missing info.
    """
    context_obj = {
        "step_index": step_index,
        "question": question,
        "answer": answer,
        "sources": [
            {"id": sid, "content": doc}
            for sid, doc in zip(source_ids, source_docs)
        ],
    }

    context_json = json.dumps(context_obj, ensure_ascii=False)

    SYSTEM_EVAL = """
You are a careful evaluation assistant.

You are given:
- A step QUESTION
- An ANSWER generated by an AI assistant
- A list of SOURCE documents (id + content) used by the assistant

Your job:
- Compare the ANSWER against the SOURCE contents.
- Determine if the answer is well supported by the sources.
- Identify hallucination risk and missing/ambiguous information.

You MUST respond with a single valid JSON object only, no extra text.
JSON schema (no comments):

{
  "step_index": int,
  "supported_by_sources": true or false,
  "hallucination_risk": "low" | "medium" | "high",
  "missing_or_ambiguous_info": true or false,
  "overall_quality": "good" | "ok" | "poor",
  "comments": "short explanation"
}
"""

    user_input = (
        "Here is the data for one retrieval step, in JSON.\n\n"
        f"{context_json}\n\n"
        "Now evaluate the answer strictly based on the sources."
    )

    resp = client.responses.parse(
        model="gpt-5-mini",
        instructions=SYSTEM_EVAL,
        input=user_input,
        text_format=StepEvaluation,
    )

    raw = resp.output_text
    data = json.loads(raw)

    usage = getattr(resp, "usage", None)
    usage_dict = {
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage and hasattr(usage, "total_tokens") else (
            (getattr(usage, "input_tokens", 0) or 0)
            + (getattr(usage, "output_tokens", 0) or 0)
            if usage else 0
        ),
    }

    try:
        eval_model = StepEvaluation.model_validate(data)
    except ValidationError as e:
        print("StepEvaluation validation error:\n", e)
        raise

    return eval_model, usage_dict


def synthesize_research_report(
    goal: str,
    plan: Plan,
    step_results: List[Dict[str, Any]],
) -> ResearchReport:
    """
    Call the LLM to turn step-wise RAG outputs + evaluations
    into a structured research report.
    """
    context_obj = {
        "goal": goal,
        "plan": plan.model_dump(),
        "steps": step_results,  # includes evaluations per step
    }

    context_json = json.dumps(context_obj, ensure_ascii=False)

    SYSTEM_REPORT = """
You are a senior research assistant.

You are given:
- A research GOAL
- A PLAN (steps with dependencies)
- For each step, a natural language answer,
  a list of source document ids,
  and an evaluation object indicating support and hallucination risk.

Your job:
- Synthesize a concise, well-structured research report.
- Base your report ONLY on the provided step answers and their evaluations.
- If evaluations show high hallucination risk or missing info,
  reflect that in the caveats and avoid presenting those parts as certain.
- Be explicit about uncertainties and limitations.

You MUST respond with a single valid JSON object that matches this schema
(do not include comments):

{
  "topic": "string - the research topic",
  "overview": "string - high-level overview (3–6 sentences)",
  "key_findings": ["string", "..."],
  "sections": [
    {
      "title": "string",
      "summary": "string",
      "bullet_points": ["string", "..."]
    }
  ],
  "caveats": ["string", "..."],
  "sources": ["doc_id_1", "doc_id_2", "..."]
}
"""

    user_input = (
        "Here is the structured intermediate data from a RAG pipeline "
        "with evaluations for each step.\n\n"
        "CONTEXT_JSON:\n"
        f"{context_json}\n\n"
        "Now synthesize the final research report as JSON matching the schema."
    )

    resp = client.responses.parse(
        model="gpt-5-mini",
        instructions=SYSTEM_REPORT,
        input=user_input,
        text_format=ResearchReport,
    )

    usage = getattr(resp, "usage", None)
    if usage:
        print(
            f"[REPORT] input_tokens={getattr(usage, 'input_tokens', 0)}, "
            f"output_tokens={getattr(usage, 'output_tokens', 0)}, "
            f"total_tokens={getattr(usage, 'total_tokens', 0)}"
        )

    raw = resp.output_text
    data = json.loads(raw)

    try:
        return ResearchReport.model_validate(data)
    except ValidationError as e:
        print("ResearchReport validation error:\n", e)
        raise


if __name__ == "__main__":
    goal = "Using the salary samples in my knowledge base, summarize key patterns and compute the average, min, max, and range."
    #goal = (
    #    "Using the salary samples in my knowledge base, summarize key patterns and compute the average, min, max, and range."
    #)
    #"Research AI Engineer job responsibilities and summarize the top 5 skills "
    #   "required, with brief descriptions."
    result = run_research_pipeline(goal)

    print("\n=== FINAL RESEARCH REPORT (STRUCTURED JSON) ===")
    print(json.dumps(result["report"], indent=2))

    print("\n=== STEP-BY-STEP EVALUATIONS ===")
    for step in result["steps"]:
        print(f"\nStep {step['step_index']}: {step['description']}")
        print("Evaluation:", json.dumps(step["evaluation"], indent=2))