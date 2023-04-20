from typing import Optional

from pydantic import BaseModel


class ClassificationCSVExample(BaseModel):
    statement: str
    prompt: str
    classes: list[str]
    answer_index: int
    formatter: str
    user_belief_raw_string: str  # empty string if no user belief
    user_belief_answer_idx: Optional[int] # None if no user belief


class ClassificationCSVResult(BaseModel):
    loss: float
    correct: int
    predicted: str
    total_logprob: float
    ground_truth: str
    user_belief: str
    user_belief_idx: Optional[int]
    user_belief_matches_predicted: Optional[bool]

