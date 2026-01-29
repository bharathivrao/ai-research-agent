# day1_planner.py
from typing import List, Optional
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()  # Load environment variables from .env file

client = OpenAI()


class PlanStep(BaseModel):
    description: str = Field(..., description="One atomic step in the plan")
    depends_on: List[int] = Field(
        default_factory=list,
        description="0-based indices of previous steps this step depends on",
    )
    tool: str = Field(
        ...,
        description="The tool to use for this step: 'rag' for knowledge retrieval, 'code' for calculations / small numeric analysis.",
    )


class Plan(BaseModel):
    goal: str
    priority: str  # "high" | "medium" | "low"
    steps: List[PlanStep]


SYSTEM_INSTRUCTIONS = """
You are a structured planning agent.

You MUST respond with a single valid JSON object only, no extra text.
JSON schema:

{
  "goal": "string",
  "priority": "string - one of: high, medium, low",
  "steps": [
    {
      "description": "string - one concrete step",
      "depends_on": [0, 1, 2],
      "tool": "string - 'rag' or 'code'"
    }
  ]
}

Tool selection rules (STRICT):
- Use "rag" for research, definitions, extraction from documents, comparisons, summarization, or anything that requires reading the knowledge base.
- Use "code" ONLY when the step clearly requires calculation, simple statistics, aggregation, transforming lists/numbers, or validating computations.
- If the step is partly research and partly computation, split it into TWO steps:
  (1) rag: gather the needed numbers/data points
  (2) code: compute the result from those numbers
- Do NOT use "code" for general reasoning, planning, or writing.
- Prefer 4–8 steps.
"""


def make_plan(goal: str) -> Plan:
    # Ask the model for JSON only
    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTIONS
            },
            {
                "role": "user",
                "content": f"Create a plan for this goal:\n{goal}"
            },
        ],
        # Force JSON output mode
        text_format=Plan
    )

    usage = getattr(response, "usage", None)
    if usage:
        print(
            f"[REPORT] input_tokens={getattr(usage, 'input_tokens', 0)}, "
            f"output_tokens={getattr(usage, 'output_tokens', 0)}, "
            f"total_tokens={getattr(usage, 'total_tokens', 0)}"
        )

    # SDK gives us aggregated text here
    raw = response.output_text

    # Debug helper while developing:
    # print("RAW MODEL OUTPUT:\n", raw)

    data = json.loads(raw)   
    # Validate & coerce into our Plan model
    try:
        return Plan.model_validate(data)
    except ValidationError as e:
        print("Pydantic validation error:\n", e)
        # You can decide how to handle this; for now just re-raise
        raise


if __name__ == "__main__":
    user_goal = "Compare LLM orchestration frameworks for building agentic systems"
    plan = make_plan(user_goal)
    # Pretty print
    print(json.dumps(plan.model_dump(), indent=2))