from data_models import ProfEvaluatorState
from agents.rmp_agent import *

def orchestrator(professor: str, class_name: str, rmp_text: str):
    state = ProfEvaluatorState(
        professor_name=professor,
        class_name=class_name
    )

    try:
        rmp_results = run_rmp_agent("")
        state.rmp_analysis = rmp_results
    except Exception as e:
        print(e)

    return state
