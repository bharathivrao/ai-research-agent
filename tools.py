# tools.py

from typing import Any, Dict
import math

from openai import OpenAI

client = OpenAI()


ALLOWED_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}

ALLOWED_MODULES = {
    "math": math,
}


def _safe_eval(expr: str) -> Any:
    """
    Safely evaluate a simple Python expression for numeric calculations.
    No access to globals beyond a small whitelist.
    """
    env = {}
    env.update(ALLOWED_BUILTINS)
    env.update(ALLOWED_MODULES)
    return eval(expr, {"__builtins__": {}}, env)


def run_python_tool(instruction: str) -> Dict[str, Any]:
    """
    Use the LLM to turn a natural language instruction into a pure Python
    expression, then safely evaluate it.

    Returns dict with:
      - expression (str)
      - result (any)
      - usage (token info)
    """
    SYSTEM_TOOL = """
    You are a Python expression generator for a calculation tool.

    You will receive:
    - A MEMORY_JSON that may contain numbers embedded in text.
    - A TASK that says what to compute.

    You must:
    - Extract the needed numbers from MEMORY_JSON (from the text).
    - Output ONLY a single valid Python expression on one line.
    - Use only arithmetic/operators and the 'math' module if needed.
    - Do NOT use variables, assignments, imports, or multi-line code.

    Examples of allowed output:
    (185000 + 210000 + 195000) / 3
    max([1, 2, 3]) - min([1, 2, 3])
    (min([185000,210000]) , max([185000,210000]))
    """

    resp = client.responses.create(
        model="gpt-5-mini",
        instructions=SYSTEM_TOOL,
        input=instruction,
    )

    expr = resp.output_text.strip()

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
        result = _safe_eval(expr)
        status = "ok"
        error = None
    except Exception as e:
        result = None
        status = "error"
        error = str(e)

    return {
        "expression": expr,
        "result": result,
        "status": status,
        "error": error,
        "usage": usage_dict,
    }