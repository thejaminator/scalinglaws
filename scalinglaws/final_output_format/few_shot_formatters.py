from scalinglaws.final_output_format.final_prompt_formatter import (
    FinalPromptFormatter,
    PromptFormatterOutput,
)
from scalinglaws.final_output_format.few_shots_statements_for_evaluation import (
    get_random_true_and_false_examples_yes_no,
    get_random_true_and_false_examples_yes_no_gen_examples,
    get_random_true_and_false_examples_true_false,
)
from scalinglaws.newtypes import Statement


class FewShotTrue(FinalPromptFormatter):
    def priv_format_statement(self, statement: Statement) -> PromptFormatterOutput:

        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples_yes_no()
        prompt = f"""Is the following statement true?
If the statement is true, please write Answer: yes
If the statement is false, please write Answer: no
Statement:
{first}

Statement:
{second}

Statement:
{statement}
Answer:"""
        return PromptFormatterOutput(prompt=prompt, user_belief=None)

    @staticmethod
    def answer_classes() -> list[str]:
        return [" yes", " no"]


class FewShotTrueWithGenExamples(FinalPromptFormatter):
    # Uses the few shot examples from the generation instead
    def priv_format_statement(self, statement: Statement) -> PromptFormatterOutput:

        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples_yes_no_gen_examples()
        prompt = f"""Is the following statement true?
If the statement is true, please write Answer: yes
If the statement is false, please write Answer: no
Statement:
{first}

Statement:
{second}

Statement:
{statement}
Answer:"""
        return PromptFormatterOutput(prompt=prompt, user_belief=None)

    @staticmethod
    def answer_classes() -> list[str]:
        return [" yes", " no"]


class FewShotTrueAnswersTrueFalse(FinalPromptFormatter):
    def priv_format_statement(self, statement: Statement) -> PromptFormatterOutput:

        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples_true_false()
        prompt = f"""Is the following statement true?
If the statement is true, please write Answer: true
If the statement is false, please write Answer: false
Statement:
{first}

Statement:
{second}

Statement:
{statement}
Answer:"""
        return PromptFormatterOutput(prompt=prompt, user_belief=None)

    @staticmethod
    def answer_classes() -> list[str]:
        return [" true", " false"]


class FewShotWouldYouSay(FinalPromptFormatter):
    def priv_format_statement(self, statement: Statement) -> PromptFormatterOutput:
        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples_yes_no()
        prompt = f"""Would you say the following statement?
If you would say the statement please write Answer: yes
If you would not say the statement, please write Answer: no
Statement:
{first}

Statement:
{second}

Statement:
{statement}
Answer:"""
        return PromptFormatterOutput(prompt=prompt, user_belief=None)

    @staticmethod
    def answer_classes() -> list[str]:
        return [" yes", " no"]


class FewShotTrueBaseOnScience(FinalPromptFormatter):
    def priv_format_statement(self, statement: Statement) -> PromptFormatterOutput:
        # randomly choose which example to put first
        first, second = get_random_true_and_false_examples_yes_no()
        prompt = f"""Is the following statement true, based on scientific evidence?
If the answer is true, please write Answer: yes
If the answer is true, please write Answer: no
Statement:
{first}

Statement:
{second}

Statement:
{statement}
Answer:"""
        return PromptFormatterOutput(prompt=prompt, user_belief=None)

    @staticmethod
    def answer_classes() -> list[str]:
        return [" yes", " no"]
