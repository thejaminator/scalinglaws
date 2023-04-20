from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel

from scalinglaws.newtypes import Statement


class UserBelief(BaseModel):
    """
    For sycophancy investigation, we add in the user belief in the prompt
    if None, then we didn't use the user belief in the prompt
    answer_idx: the index of the user belief in the answer_classes
    raw_String: the raw user belief string
    """
    answer_idx: Literal[0, 1]
    raw_string: str
class PromptFormatterOutput(BaseModel):
    prompt: str
    user_belief: Optional[UserBelief]


class FinalPromptFormatter(ABC):
    def __priv_format_statement(self, statement: Statement) -> PromptFormatterOutput:
        self_name = self.__class__.__name__
        raise NotImplementedError(f"__format_statement not implemented for {self_name}")

    def format_statement_with_ground_truth(
        self, statement: Statement, ground_truth: bool
    ) -> PromptFormatterOutput:
        # by default, we just format the statement
        return self.__priv_format_statement(statement)

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def all_formatters(cls) -> list["FinalPromptFormatter"]:
        return [cls() for cls in cls.__subclasses__()]

    @classmethod
    def formatter_path(cls) -> Path:
        return Path("data") / cls.name()

    @staticmethod
    @abstractmethod
    def answer_classes() -> list[str]:
        raise NotImplementedError
