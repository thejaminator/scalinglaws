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
