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

class SchoolCandidate(BaseModel):
    id: str
    name: str
    city: str
    state: str

class SchoolSelection(BaseModel):
    selected_id: str
    confidence_score: float # 0.0 to 1.0
    reasoning: str

class ProfEvaluatorState(BaseModel):
    """
    A data model for the state of the orchestrator.
    """
    professor_name: str
    class_name: str
    school_query: Optional[str] = None
    selected_school_id: Optional[str] = None
    school_selection_confidence: Optional[float] = None
    school_selection_reasoning: Optional[str] = None
    ambiguity_prompt: Optional[str] = None
    rmp_analysis: Optional[RMPData] = None
    final_result: Optional[str] = None
