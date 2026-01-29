# planner_rag_agent.py

from planner import make_plan
from rag_chroma import answer_with_rag
import json


def run_agent(goal: str):
    # 1) PLANNING
    plan = make_plan(goal)

    print("\n=== PLAN ===")
    print(json.dumps(plan.model_dump(), indent=2))

    results = []

    # 2) EXECUTE EACH STEP USING RAG
    print("\n=== EXECUTION ===")
    for idx, step in enumerate(plan.steps):
        print(f"\nStep {idx}: {step.description}")

        answer, sources = answer_with_rag(step.description)

        results.append({
            "step": idx,
            "description": step.description,
            "answer": answer,
            "sources": sources,
        })

        print("Answer:")
        print(answer)
        print("Sources:", sources)

    # 3) FINAL SYNTHESIS (NAIVE)
    print("\n=== FINAL OUTPUT (DRAFT) ===")
    for r in results:
        print(f"- Step {r['step']}: {r['answer']}")

    return {
        "goal": goal,
        "plan": plan.model_dump(),
        "results": results,
    }


if __name__ == "__main__":
    goal = "Research what skills an AI engineer needs."
    run_agent(goal)