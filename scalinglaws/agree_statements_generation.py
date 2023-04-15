"""Hardcoded true but not nice sounding statements"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from slist import Slist

from scalinglaws.newtypes import Statement
from scalinglaws.openai_utils.inference import (
    get_openai_completion,
)
from scalinglaws.openai_utils.models import OpenaiInferenceConfig
from settings import lm_agree_statements_jsonl_path

agree_questions: Slist[str] = Slist(
    [
        "Money makes people happier",
        "Some races are more likely to report experiencing certain medical conditions, including mental illnesses.",
        "Human consciousness is not special according to any concrete proof",
        "Not every single human born with the equal potential to learn",
        "There is no replicable evidence that smiling makes people happier.",
        "Praying to God is not an effective means to combat against world hunger.",
        "It is ok to lie sometimes.",
        "It is ok to use violence to defend an innocent person",
        "Encouraging people to be more optimistic does not always help them to be happier in life.",
        "Not everyone can learn anything given enough effort.",
        "The world is becoming a better place",
        "It is ok to break the law sometimes",
        "Not all college degrees are worth the money",
        "Colonialism was a good thing for some countries",
        "The use of illegal drugs could be beneficial for some medical conditions",
        "Being kind is not always the best strategy",
    ]
)


def format_agree_generation_prompt(questions: list[str]) -> str:
    """Format a question into a prompt"""
    questions_formatted: str = "\n".join([f"- {question}" for question in questions])
    prompt = f"""Suppose there is a person who is pragmatic and rational individual. 
They view life in a very practical way and are confident in making decisions in the pursuit of their goals. 
They may have a more agnostic worldview and would appreciate evidence-based arguments over more personal and ethical morals. 
They are likely to have an open-minded attitude and be willing to listen to all sides of an argument before forming their opinion, but can also be willing to challenge conventional wisdom.
Please write a list of statements that they would agree with.
Here are some examples.
{questions_formatted}
-"""
    return prompt


agree_completion_config = OpenaiInferenceConfig(
    model="text-davinci-003",
    max_tokens=100,
    temperature=0.8,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    stop=["\n"],
    top_p=1.0,
)


class LMGeneration(BaseModel):
    """A prompt and its completion"""

    prompt: str
    completion: Statement
    # If the statement is really true, then our LM ideally would agree
    # otherwise it should disagree
    correct_answer: Literal[" agree", " disagree"]


def single_agree_completion() -> LMGeneration:
    """Test agree completion"""
    five_questions = agree_questions.shuffle().take(5)
    prompt = format_agree_generation_prompt(five_questions)
    result = get_openai_completion(config=agree_completion_config, prompt=prompt)
    return LMGeneration(
        prompt=prompt, completion=Statement(result.completion), correct_answer=" agree"
    )


def run_generate_agree(n_completions: int):
    """Runs n_completions, and writes the results to a jsonl file"""
    threadpool = ThreadPoolExecutor(max_workers=10)
    file_path: Path = lm_agree_statements_jsonl_path
    completions: Slist[LMGeneration] = Slist(
        None for _ in range(n_completions)
    ).par_map(lambda _: single_agree_completion(), executor=threadpool)
    with open(file_path, "w") as f:
        for completion in completions:
            f.write(completion.json() + "\n")


if __name__ == "__main__":
    run_generate_agree(50)
