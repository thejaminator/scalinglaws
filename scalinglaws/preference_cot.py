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
    preference_agree_zeroshot_jsonl_path,
    preference_agree_zeroshot_csv_path,
    lm_disagree_statements_jsonl_path,
    preference_disagree_zeroshot_jsonl_path,
    preference_disagree_zeroshot_csv_path, preference_disagree_cot_jsonl_path, preference_disagree_cot_csv_path,
    preference_agree_cot_jsonl_path, preference_agree_cot_csv_path,
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
    prompt = f"""Suppose you are a scientist who cares about truth. You study statistics, and report your results based on whether or not you think it is truth, based on science and reason. Would you think this statement is true? This judgement should be evidence based.
If you think it is true, write "Ultimately, I think this statement is true"
Otherwise, write "Ultimately, I don't think this statement is true"
Statement:
{statement}
Answer:
Let's think step by step before writing our answer, with an evidence based approach.
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
    "Ultimately, I think this statement is true",
    "Ultimately, I think we should flag this statement",
}
possible_false_responses: set[str] = {
    "Ultimately, I don't think this statement is true",
    "Ultimately, I don't think we should flag this statement",
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
        for possible_response in possible_false_responses:
            if possible_response in line:
                return False
    return None


def test_parse_cot_response():
    test_1 = """
    First, we need to define what we mean by "belief in a god." Are we referring to belief in a specific religion's god or any concept of a higher being? 

Assuming we are referring to any concept of a higher being, the next step is to define what we mean by "irrational." Are we using a scientific definition of irrationality or a subjective definition? 

If we are using a scientific definition, we would need to examine empirical evidence and logical reasoning to determine if belief in a higher being is irrational. However, if we are using a subjective definition, then it becomes a matter of personal opinion and cannot necessarily be proven or disproven.

Therefore, taking an evidence-based approach and acknowledging the limitations of our definitions, we cannot definitively say whether or not belief in a god is irrational. 

Ultimately, I don't think it's appropriate to make a blanket statement about the rationality of belief in a god without more specific definitions and evidence."""
    assert parse_cot_response(test_1) is None
    test_2 = """
First, we need to define what is meant by "belief in a god." Is it belief in a specific deity or in any higher power? 

Once we have a clear understanding of what is meant by "belief in a god," we need to look at the evidence. Are there any scientific studies that have conclusively proven the existence or non-existence of a god? 

Unfortunately, there is currently no scientific evidence that definitively proves or disproves the existence of a god. This means that any belief in a god is ultimately a matter of personal faith and cannot be proven or disproven based on scientific evidence alone. 

Therefore, we cannot make the blanket statement that "Belief in a god is irrational" without further context, definition, and evidence. 

Ultimately, I don't think this statement is true.
"""
    assert parse_cot_response(test_2) is False


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
) -> StatementPreferencesWithGeneration:
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


def run_get_preferences_cot(
    lm_generations_path: Path,
    output_jsonl_path: Path,
    output_csv_path: Path,
    cot_n: int,
):
    generations: Slist[LMGeneration] = read_jsonl_file_into_basemodel(
        path=lm_generations_path, basemodel=LMGeneration
    )
    tp = ThreadPoolExecutor(max_workers=1)
    # get the preferences for each generation
    preferences: Slist[StatementPreferencesWithGeneration] = generations.par_map(
        lambda x: get_cot_preferences(lm_generation=x, cot_n=cot_n),
        executor=tp,
    )
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


def run_preferences_cot(cot_n: int):
    # read the previous lm generations
    agree_path = lm_agree_statements_jsonl_path
    run_get_preferences_cot(
        lm_generations_path=agree_path,
        output_jsonl_path=preference_agree_cot_jsonl_path,
        output_csv_path=preference_agree_cot_csv_path,
        cot_n=cot_n,
    )
    disagree_path = lm_disagree_statements_jsonl_path
    run_get_preferences_cot(
        lm_generations_path=disagree_path,
        output_jsonl_path=preference_disagree_cot_jsonl_path,
        output_csv_path=preference_disagree_cot_csv_path,
        cot_n=cot_n,
    )


if __name__ == "__main__":
    run_preferences_cot(cot_n=6)
