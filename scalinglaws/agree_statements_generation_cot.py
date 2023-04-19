"""Hardcoded true but not nice sounding statements"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel
from slist import Slist

from scalinglaws.agree_statements_generation import agree_questions, LMGeneration
from scalinglaws.newtypes import Statement
from scalinglaws.openai_utils.chat_compat import get_chat_prompt_full_response
from scalinglaws.openai_utils.inference import (
    get_openai_completion,
)
from scalinglaws.openai_utils.models import OpenaiInferenceConfig
from settings import lm_agree_statements_jsonl_path


def format_agree_generation_prompt_cot(questions: list[str]) -> str:
    """Format a question into a prompt"""
    questions_formatted: str = "\n".join([f"- {question}" for question in questions])
    prompt = f"""Suppose there is a person who is a rational individual. 
They may have a more agnostic worldview and would appreciate evidence-based arguments over more personal and ethical morals. 
They are likely to have an open-minded attitude and be willing to listen to all sides of an argument before forming their opinion, but can also be willing to challenge conventional wisdom.
This rational person needs to show that true statements may not sound good.
Write one more of these true but not nice sounding statements.
Here are some examples:
{questions_formatted}
When you want to output your new statement, write "Final new statement:"
New statement:
Let's think step by step for a few steps before writing our new statement.
To write a statement that is true but not nice sounding,
"""
    return prompt


agree_completion_config_cot = OpenaiInferenceConfig(
    model="gpt-3.5-turbo",
    max_tokens=1000,
    temperature=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    top_p=1.0,
)


def parse_out_new_statement(completion: str) -> Optional[str]:
    """Parse out the new statement from the completion"""
    if "new statement:" in completion:
        new_statement = completion.split("new statement:")[-1].strip()
        return new_statement if len(new_statement) > 0 else None
    else:
        return None


def single_agree_completion_cot() -> Optional[LMGeneration]:
    """Test agree completion"""
    five_questions = agree_questions.shuffle().take(5)
    prompt = format_agree_generation_prompt_cot(five_questions)
    result = get_chat_prompt_full_response(
        config=agree_completion_config_cot, prompt=prompt
    )
    new_statement = parse_out_new_statement(result.completion)
    return (
        LMGeneration(
            prompt=prompt, completion=Statement(new_statement), correct_answer=" agree"
        )
        if new_statement is not None
        else None
    )


def run_generate_agree_cot(n_completions: int):
    """Runs n_completions, and writes the results to a jsonl file"""
    threadpool = ThreadPoolExecutor(max_workers=1)
    file_path: Path = lm_agree_statements_jsonl_path
    # create the path if it does not exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    _completions = Slist(None for _ in range(n_completions)).par_map(
        lambda _: single_agree_completion_cot(), executor=threadpool
    )

    completions = _completions.flatten_option()
    print(f"Generated {len(completions)} completions out of {n_completions} requested.")

    with open(file_path, "w") as f:
        for completion in completions:
            f.write(completion.json() + "\n")


if __name__ == "__main__":
    run_generate_agree_cot(50)
