from typing import Optional

from pydantic import BaseModel, validator
import math

class ClassificationCSVExample(BaseModel):
    statement: str
    prompt: str
    classes: list[str]
    answer_index: int
    formatter: str
    user_belief_raw_string: str  # empty string if no user belief
    user_belief_answer_idx: Optional[int] # None if no user belief

    """dumbs hacks cos of CSVs"""
    @validator("user_belief_answer_idx", pre=True)
    def convert_empty_string_to_none(cls, v):
        if isinstance(v, str):
            if v == "":
                return None
            else:
                return v
        if v is None:
            return None
        if math.isnan(v):
            return None
        return v

    # convert nan to None for user_belief_raw_string
    @validator("user_belief_raw_string", pre=True)
    def convert_nan_to_empty(cls, v):
        if isinstance(v, str):
            return v
        if math.isnan(v):
            return ""
        return v



class ClassificationCSVResult(BaseModel):
    loss: float
    correct: int
    predicted: str
    total_logprob: float
    ground_truth: str
    user_belief: str
    user_belief_idx: Optional[int]
    user_belief_matches_predicted: Optional[bool]

