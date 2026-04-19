from data_models import ProfEvaluatorState
from agents.rmp_agent import *

def orchestrator(professor: str, class_name: str, rmp_text: str, school_query: str | None = None):
    state = ProfEvaluatorState(
        professor_name=professor,
        class_name=class_name,
        school_query=school_query
    )

    if school_query:
        school_resolution = resolve_school_query(school_query)
        status = school_resolution["status"]

        if "confidence_score" in school_resolution:
            state.school_selection_confidence = school_resolution["confidence_score"]
        if "reasoning" in school_resolution:
            state.school_selection_reasoning = school_resolution["reasoning"]

        if status in ("high_confidence", "medium_confidence"):
            state.selected_school_id = school_resolution["selected_id"]
        elif status == "low_confidence":
            state.ambiguity_prompt = school_resolution["message"]
            return state
        elif status == "not_found":
            state.ambiguity_prompt = school_resolution["message"]
            return state

    try:
        rmp_results = run_rmp_agent(rmp_text)
        state.rmp_analysis = rmp_results
    except Exception as e:
        print(e)

    return state
