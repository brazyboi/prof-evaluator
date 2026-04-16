from typing import Optional
from pydantic import BaseModel

class RMPData(BaseModel):
    """
    The data model for what the RMP agent is going to return.
    """
    average_rating: float
    difficulty_score: float
    top_tags: list[str]
    confidence_score: float # this is the weight we apply based on # of reviews

class ProfEvaluatorState(BaseModel):
    """
    A data model for the state of the orchestrator.
    """
    professor_name: str
    class_name: str
    rmp_analysis: Optional[RMPData] = None
    final_result: Optional[str] = None
