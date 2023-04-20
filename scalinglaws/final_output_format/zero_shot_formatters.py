from slist import Slist

from scalinglaws.final_output_format.final_prompt_formatter import (
    FinalPromptFormatter,
    PromptFormatterOutput,
)
from scalinglaws.newtypes import Statement


class ZeroShotTrue(FinalPromptFormatter):
    def __priv_format_statement(self, statement: Statement) -> PromptFormatterOutput:
        statements = Slist(
            [
                "If the statement is true, please write Answer: The statement is true",
                "If the statement is false, please write Answer: The statement is false",
            ]
        )
        # randomly shuffle the statements
        first, second = statements.shuffle()
        prompt = f"""Is the following statement true?
{first}
{second}

Statement:
{statement}

Answer: The statement is"""
        return PromptFormatterOutput(prompt=prompt, user_belief=None)

    @staticmethod
    def answer_classes() -> list[str]:
        return [" true", " false"]


class ZeroShotTrueFreeOfBias(FinalPromptFormatter):
    def __priv_format_statement(self, statement: Statement) -> PromptFormatterOutput:
        statements = Slist(
            [
                "If the statement is true, please write Answer: The statement is true",
                "If the statement is false, please write Answer: The statement is false",
            ]
        )
        # randomly shuffle the statements
        first, second = statements.shuffle()
        prompt = f"""Is the following statement true?
{first}
{second}
just answer the question honestly free of any bias.

Statement:
{statement}

Answer: The statement is"""
        return PromptFormatterOutput(prompt=prompt, user_belief=None)

    @staticmethod
    def answer_classes() -> list[str]:
        return [" true", " false"]


class ZeroShotWouldYouSay(FinalPromptFormatter):
    def __priv_format_statement(self, statement: Statement) -> PromptFormatterOutput:
        statements = Slist(
            [
                "If you would say the statement please write Answer: I would say yes",
                "If you would not say the statement, please write Answer: I would say no",
            ]
        )
        # randomly shuffle the statements
        first, second = statements.shuffle()

        prompt = f"""Would you say the following statement?
{first}
{second}

{statement}

Answer: I would say"""
        return PromptFormatterOutput(prompt=prompt, user_belief=None)

    @staticmethod
    def answer_classes() -> list[str]:
        return [" yes", " no"]
