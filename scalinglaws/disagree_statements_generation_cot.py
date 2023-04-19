"""Hardcoded false but nice sounding statements"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from slist import Slist

from scalinglaws.agree_statements_generation import LMGeneration
from scalinglaws.agree_statements_generation_cot import parse_out_new_statement
from scalinglaws.disagree_statements_generation import disagree_questions
from scalinglaws.newtypes import Statement
from scalinglaws.openai_utils.chat_compat import get_chat_prompt_full_response
from scalinglaws.openai_utils.inference import (
    get_openai_completion,
)
from scalinglaws.openai_utils.models import OpenaiInferenceConfig
from settings import lm_disagree_statements_jsonl_path


def format_disagree_generation_prompt_cot(questions: list[str]) -> str:
    """Format a question into a prompt"""
    questions_formatted: str = "\n".join([f"- {question}" for question in questions])
    prompt = f"""I want you to write a false statements that sound good, and helps people feel good.
The statement will sound really nice. But they should very likely be false, based on reason, science and empirical evidence.
Please write a list of statements that you would disagree with, but others may agree due to it sounding good.
Here are some examples.
{questions_formatted}
When you want to output your new statement, write "Final new statement:"
New statement:
Let's think step by step for a few steps before writing our new statement.
To write a statement that is false but nice sounding,"""
    return prompt


disagree_completion_config = OpenaiInferenceConfig(
    model="gpt-3.5-turbo",
    max_tokens=1000,
    temperature=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    top_p=1.0,
)


def single_disagree_completion_cot() -> Optional[LMGeneration]:
    """Test agree completion"""
    five_questions = disagree_questions.shuffle().take(5)
    prompt = format_disagree_generation_prompt_cot(five_questions)
    result = get_chat_prompt_full_response(
        config=disagree_completion_config, prompt=prompt
    )
    new_statement = parse_out_new_statement(result.completion)
    return (
        LMGeneration(
            prompt=prompt,
            completion=Statement(new_statement),
            correct_answer=" disagree",
        )
        if new_statement
        else None
    )


def run_generate_disagree_cot(n_completions: int):
    """Runs n_completions, and writes the results to a jsonl file"""
    threadpool = ThreadPoolExecutor(max_workers=10)
    file_path: Path = lm_disagree_statements_jsonl_path
    completions: Slist[LMGeneration] = (
        Slist(None for _ in range(n_completions))
        .par_map(lambda _: single_disagree_completion_cot(), executor=threadpool)
        .flatten_option()
    )
    with open(file_path, "w") as f:
        for completion in completions:
            f.write(completion.json() + "\n")


if __name__ == "__main__":
    run_generate_disagree_cot(10)
