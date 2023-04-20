from abc import ABC, abstractmethod
from pathlib import Path

from scalinglaws.newtypes import Statement


class FinalPromptFormatter(ABC):
    def private_format_statement(self, statement: Statement) -> str:
        self_name = self.__class__.__name__
        raise NotImplementedError(f"__format_statement not implemented for {self_name}")

    def format_statement_with_ground_truth(
        self, statement: Statement, ground_truth: bool
    ) -> str:
        # by default, we just format the statement
        return self.private_format_statement(statement)

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
