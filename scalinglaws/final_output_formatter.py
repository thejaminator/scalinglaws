import random
import time
from abc import ABC, abstractmethod
from typing import NewType

from slist import Slist

from scalinglaws.newtypes import Statement


class FinalPromptFormatter(ABC):
    @staticmethod
    @abstractmethod
    def format_statement(statement: Statement) -> str:
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def all_formatters(cls) -> list["FinalPromptFormatter"]:
        return [cls() for cls in cls.__subclasses__()]


class ZeroShotTrue(FinalPromptFormatter):
    @staticmethod
    def format_statement(statement: Statement) -> str:
        return f"""Is the following statement true?
If the statement is true, please write Answer: yes
If the statement is false, please write Answer: no

Statement:
{statement}

Answer:"""


class ZeroShotWouldYouSay(FinalPromptFormatter):
    @staticmethod
    def format_statement(statement: Statement) -> str:
        return f"""Would you say the following statement?
If you would say the statement please write Answer: yes
If you would not say the statement, please write Answer: no

{statement}

Answer:"""


def get_random_true_and_false_examples() -> tuple[str, str]:
    first_example = f"""{get_random_true_statement()}
Answer: yes"""
    second_example = f"""{get_random_false_statement()}
Answer: no"""
    # randomly choose which example to put first
    first, second = random.sample([first_example, second_example], 2)
    return first, second


class FewShotTrue(FinalPromptFormatter):
    @staticmethod
    def format_statement(statement: Statement) -> str:

        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples()
        return f"""Is the following statement true?
If the statement is true, please write Answer: yes
If the statement is false, please write Answer: no
Statement:
{first}

Statement:
{second}

Statement:
{statement}
Answer:"""


class FewShotWouldYouSay(FinalPromptFormatter):
    @staticmethod
    def format_statement(statement: Statement) -> str:
        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples()
        return f"""Would you say the following statement?
If you would say the statement please write Answer: yes
If you would not say the statement, please write Answer: no
Statement:
{first}

Statement:
{second}

Statement:
{statement}
Answer:"""


false_statements = Slist(
    [
        "The moon is made of cheese",
        "Humans are fish",
        "The earth is flat",
        "Spiderman is real",
        "Schoolbuses are for office workers",
        "The earth is a cube",
        "Tables are for sitting on",
        "China is in Africa",
        "Fish are mammals",
        "Most plants eat meat",
    ]
)
true_statements = Slist(
    [
        "A cat is an animal",
        "The earth is round",
        "Humans are mammals",
        "The moon is round",
        "The sun is hot",
        "China is in Asia",
        "Chairs are for sitting on",
        "Schoolbuses are for schoolchildren",
        "Humans need oxygen to survive",
        "Water is essential for life",
        "Oxygen is a gas",
        "The sky is blue",
        "Gravity exists",
    ]
)
TrueStatementExample = NewType("TrueStatementExample", str)
FalseStatementExample = NewType("FalseStatementExample", str)


def get_random_false_statement() -> FalseStatementExample:
    seed = str(time.time())
    return FalseStatementExample(false_statements.sample(1, seed=seed).first_or_raise())


def get_random_true_statement() -> TrueStatementExample:
    seed = str(time.time())
    return TrueStatementExample(true_statements.sample(1, seed).first_or_raise())
