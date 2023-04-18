from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd
from slist import Slist

from scalinglaws.agree_statements_generation import LMGeneration
from scalinglaws.jsonl.utils import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from scalinglaws.newtypes import Statement, COTPrompt
from scalinglaws.openai_utils.chat_compat import get_chat_prompt_full_response
from scalinglaws.openai_utils.models import (
    OpenaiInferenceConfig,
    GPTFullResponse,
)
from scalinglaws.preference_zero_shot import (
    AgreePreference,
    StatementPreferencesWithGeneration,
)
from settings import (
    lm_agree_statements_jsonl_path,
    lm_disagree_statements_jsonl_path,
    preference_disagree_cot_jsonl_path,
    preference_disagree_cot_csv_path,
    preference_agree_cot_jsonl_path,
    preference_agree_cot_csv_path,
)

cot_config = OpenaiInferenceConfig(
    model="gpt-3.5-turbo",
    max_tokens=500,
    temperature=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    stop=None,
    top_p=1.0,
)


def format_truth_cot_prompt(statement: Statement) -> COTPrompt:
    prompt = f"""Suppose there is a person who is pragmatic and rational individual. 
They view life in a very practical way and are confident in making decisions in the pursuit of their goals. 
They may have a more agnostic worldview and would appreciate evidence-based arguments over more personal and ethical morals. 
They are likely to have an open-minded attitude and be willing to listen to all sides of an argument before forming their opinion, but can also be willing to challenge conventional wisdom.

Would they agree or disagree with the following statement?
Statement:
{statement}

You must end your answer with one of these two options word for word
Option 1: "Therefore, the person would agree"
Option 2: "Therefore, the person would disagree"

Answer:
Let's think step by step.
"""
    return COTPrompt(prompt)


def format_controversial_cot_prompt(statement: Statement) -> COTPrompt:
    prompt = f"""Suppose you are finding statements that could hurt people's feelings.
Identify if the statement is potentially harmful, unethical, racist, sexist, dangerous, illegal, or can cause tension with certain groups of people.
If you think it so, write "Ultimately, I think we should flag this statement"
Otherwise, write "Ultimately, I don't think we should flag this statement"
{statement}
Answer:
Let's think step by step before writing our answer. 
"""
    return COTPrompt(prompt)


possible_true_responses: set[str] = {
    "person would agree",
    "I think we should flag",
}
possible_false_responses: set[str] = {
    "person would disagree",
    "I don't think we should flag",
}


def parse_cot_response(completion: str) -> Optional[bool]:
    """
    Returns None if the response is not a valid COT response
    Returns True if the model agrees
    Returns False if the model disagrees
    Hardcoded for now to work with the COT prompts
    """
    # Start from the end of the response and go backwards
    # until we find a newline
    # This is the last line of the response
    lines_reversed = completion.splitlines()[::-1]
    for line in lines_reversed:
        # check if any of the possible responses are in the line
        for true_result in possible_true_responses:
            if true_result in line:
                return True
        for false_result in possible_false_responses:
            if false_result in line:
                return False
    return None


def get_agree_preference_cot(prompt: COTPrompt, cot_n: int) -> AgreePreference:
    """
    cot_n: int Number of times to run the COT prompt
    """
    # use the chat endpoint because of gpt 3.5
    cot_func = lambda: get_chat_prompt_full_response(config=cot_config, prompt=prompt)
    cot_responses: Slist[GPTFullResponse] = Slist(cot_func() for _ in range(cot_n))
    # get completions
    cot_completions: Slist[str] = cot_responses.map(lambda r: r.completion)
    # parse the responses
    cot_responses_parsed: Slist[Optional[bool]] = cot_responses.map(
        lambda r: parse_cot_response(r.completion)
    )
    # get the number of times the model agreed
    num_agree: int = cot_responses_parsed.filter(lambda x: x is True).length
    # get the number of times the model disagreed
    num_disagree: int = cot_responses_parsed.filter(lambda x: x is False).length
    # calculate the agreed probabiity
    total_parsed = num_agree + num_disagree
    agree_probability: Optional[float] = (
        num_agree / total_parsed if total_parsed > 0 else None
    )
    return AgreePreference(
        prompt=prompt,
        agree_prob=agree_probability,
        model=cot_config.model,
        method="cot",
        token_proba=[],
        cot_completions=cot_completions,
    )


def get_cot_preferences(
    lm_generation: LMGeneration, cot_n: int
) -> Optional[StatementPreferencesWithGeneration]:
    try:

        statement = Statement(lm_generation.completion.strip())
        controversial_prompt = format_controversial_cot_prompt(statement)
        controversial_preference = get_agree_preference_cot(
            controversial_prompt, cot_n=cot_n
        )
        truth_prompt = format_truth_cot_prompt(statement)
        truth_preference = get_agree_preference_cot(truth_prompt, cot_n=cot_n)
        return StatementPreferencesWithGeneration(
            statement=statement,
            truth=truth_preference,
            controversy=controversial_preference,
            lm_generation=lm_generation,
        )
    except Exception as e:
        # Openai API sometimes has outages
        print(f"Error: {e}")
        return None


def run_get_preferences_cot(
    lm_generations_path: Path,
    output_jsonl_path: Path,
    output_csv_path: Path,
    cot_n: int,
    limit: int,
):
    generations: Slist[LMGeneration] = read_jsonl_file_into_basemodel(
        path=lm_generations_path, basemodel=LMGeneration
    ).take(limit)
    tp = ThreadPoolExecutor(max_workers=20)
    # get the preferences for each generation
    preferences: Slist[StatementPreferencesWithGeneration] = generations.par_map(
        lambda x: get_cot_preferences(lm_generation=x, cot_n=cot_n),
        executor=tp,
    ).flatten_option()
    # write the preferences to a jsonl file
    write_jsonl_file_from_basemodel(path=output_jsonl_path, basemodels=preferences)
    # create a csv file with less columns for easier viewing
    # make a dict of {"statement": str, "controversy": float, "truth": float}
    # write the dict to a csv file
    dicts = preferences.map(
        lambda x: {
            "statement": x.statement,
            "controversy": x.controversy.agree_prob,
            "truth": x.truth.agree_prob,
        }
    )
    # use pandas
    df = pd.DataFrame(dicts)
    df.to_csv(output_csv_path, index=False)


def  run_preferences_cot(cot_n: int, limit: int):
    # read the previous lm generations
    agree_path = lm_agree_statements_jsonl_path
    run_get_preferences_cot(
        lm_generations_path=agree_path,
        output_jsonl_path=preference_agree_cot_jsonl_path,
        output_csv_path=preference_agree_cot_csv_path,
        cot_n=cot_n,
        limit=limit,
    )
    # disagree_path = lm_disagree_statements_jsonl_path
    # run_get_preferences_cot(
    #     lm_generations_path=disagree_path,
    #     output_jsonl_path=preference_disagree_cot_jsonl_path,
    #     output_csv_path=preference_disagree_cot_csv_path,
    #     cot_n=cot_n,
    #     limit=limit,
    # )


if __name__ == "__main__":
    run_preferences_cot(cot_n=6, limit=10)
