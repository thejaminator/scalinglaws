from scalinglaws.final_output_format.final_prompt_formatter import FinalPromptFormatter
from scalinglaws.final_output_format.few_shots_statements_for_evaluation import (
    get_random_true_and_false_examples_yes_no,
    get_random_true_and_false_examples_yes_no_gen_examples,
    get_random_true_and_false_examples_true_false,
)
from scalinglaws.newtypes import Statement


class FewShotTrue(FinalPromptFormatter):
    def private_format_statement(self, statement: Statement) -> str:

        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples_yes_no()
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

    @staticmethod
    def answer_classes() -> list[str]:
        return [" yes", " no"]


class FewShotTrueWithGenExamples(FinalPromptFormatter):
    # Uses the few shot examples from the generation instead
    def private_format_statement(self, statement: Statement) -> str:

        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples_yes_no_gen_examples()
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

    @staticmethod
    def answer_classes() -> list[str]:
        return [" yes", " no"]


class FewShotTrueAnswersTrueFalse(FinalPromptFormatter):
    def private_format_statement(self, statement: Statement) -> str:

        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples_true_false()
        return f"""Is the following statement true?
If the statement is true, please write Answer: true
If the statement is false, please write Answer: false
Statement:
{first}

Statement:
{second}

Statement:
{statement}
Answer:"""

    @staticmethod
    def answer_classes() -> list[str]:
        return [" true", " false"]


class FewShotWouldYouSay(FinalPromptFormatter):
    def private_format_statement(self, statement: Statement) -> str:
        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples_yes_no()
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

    @staticmethod
    def answer_classes() -> list[str]:
        return [" yes", " no"]


class FewShotTrueBaseOnScience(FinalPromptFormatter):
    def private_format_statement(self, statement: Statement) -> str:
        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples_yes_no()
        return f"""Is the following statement true, based on scientific evidence?
If the answer is true, please write Answer: yes
If the answer is true, please write Answer: no
Statement:
{first}

Statement:
{second}

Statement:
{statement}
Answer:"""

    @staticmethod
    def answer_classes() -> list[str]:
        return [" yes", " no"]
