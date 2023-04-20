import random
from typing import NewType

from slist import Slist

from scalinglaws.agree_statements_generation import agree_questions
from scalinglaws.disagree_statements_generation import disagree_questions

FALSE_FEW_SHOT_EVAL = Slist(
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
TRUE_FEW_SHOT_EVAL = Slist(
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


def get_random_true_and_false_examples_yes_no() -> tuple[str, str]:
    first_example = f"""{get_random_true_statement()}
Answer: yes"""
    second_example = f"""{get_random_false_statement()}
Answer: no"""
    # randomly choose which example to put first
    first, second = random.sample([first_example, second_example], 2)
    return first, second


def get_random_true_and_false_examples_true_false() -> tuple[str, str]:
    first_example = f"""{get_random_true_statement()}
Answer: true"""
    second_example = f"""{get_random_false_statement()}
Answer: false"""
    # randomly choose which example to put first
    first, second = random.sample([first_example, second_example], 2)
    return first, second


def get_random_false_statement() -> FalseStatementExample:
    return FalseStatementExample(
        FALSE_FEW_SHOT_EVAL.sample(1, seed="42").first_or_raise()
    )


def get_random_true_statement() -> TrueStatementExample:
    return TrueStatementExample(TRUE_FEW_SHOT_EVAL.sample(1, "42").first_or_raise())


"""These functions get a few shot sample from the generation statements instead"""


def get_random_true_statement_from_gen() -> TrueStatementExample:
    return TrueStatementExample(agree_questions.sample(1, "42").first_or_raise())


def get_random_false_statement_from_gen() -> FalseStatementExample:
    return FalseStatementExample(
        disagree_questions.sample(1, seed="42").first_or_raise()
    )


def get_random_true_and_false_examples_yes_no_gen_examples() -> tuple[str, str]:
    first_example = f"""{get_random_true_statement_from_gen()}
Answer: yes"""
    second_example = f"""{get_random_false_statement_from_gen()}
Answer: no"""
    # randomly choose which example to put first
    first, second = random.sample([first_example, second_example], 2)
    return first, second
