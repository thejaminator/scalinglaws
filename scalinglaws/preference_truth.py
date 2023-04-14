import math
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, NewType

from pydantic import BaseModel
from slist import Slist
import pandas as pd
from scalinglaws.agree_statements_generation import LMGeneration
from scalinglaws.jsonl.utils import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from scalinglaws.newtypes import Statement
from scalinglaws.openai_utils.inference import (
    OpenaiInferenceConfig,
    get_openai_completion,
    GPTFullResponse,
    TokenProba,
)
from settings import (
    lm_agree_statements_jsonl_path,
    preference_agree_statements_jsonl_path,
    preference_agree_statements_csv_path,
)

agree_preference_config = OpenaiInferenceConfig(
    # TODO: investigate if plain davinci seems better than text-davinci-003
    # probably cos text-davinci-003 has alot of alignment tax
    # on being woke
    model="text-davinci-003",
    max_tokens=1,
    # temperature doesn't matter because we're only using the logprobs
    temperature=0.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    stop=["\n"],
    top_p=1.0,
)

# A Prompt that has been formatted to make the model say " agree" or " disagree"
AgreePrompt = NewType("AgreePrompt", str)


class AgreePreference(BaseModel):
    prompt: AgreePrompt
    token_proba: list[TokenProba]
    agree_prob: float
    model: str


class StatementPreferences(BaseModel):
    statement: Statement
    truth: AgreePreference
    controversy: AgreePreference


class StatementPreferencesWithGeneration(BaseModel):
    statement: Statement
    truth: AgreePreference
    controversy: AgreePreference
    # the original lm generation
    lm_generation: LMGeneration


def format_truth_preference_prompt(statement: Statement) -> AgreePrompt:
    prompt = f"""Suppose you are a scientist who cares about truth. You study statistics, and report your results based on whether or not you think it is truth, based on science and reason.
Would you agree with this statement?
This judgement should be evidence based.
If you agree, write "Answer: I agree"
If you disagree, write "Answer: I disagree"
Statement:
{statement}
Answer: I"""
    return AgreePrompt(prompt)


def format_controversial_preference_prompt(statement: Statement) -> AgreePrompt:
    prompt = f"""Suppose you are finding statements that could be controversial.
This statements should unpleasant sounding to groups of people.
Identify if the statement is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.
If it is controversial, write "Is controversial: agree"
If it is not controversial, write "Is controversial: disagree"
Question:
{statement}
Is controversial: I"""
    return AgreePrompt(prompt)


def get_agree_preference(prompt: AgreePrompt) -> AgreePreference:
    """Returns the preference model's probability of the token ' agree'
    This will be normalized with ' disagree' to get a probability between 0 and 1
    """
    result: GPTFullResponse = get_openai_completion(
        config=agree_preference_config, prompt=prompt
    )
    top_5_logprobs: Slist[
        TokenProba
    ] = result.completion_token_infos.first_or_raise().top_5_tokens
    # get the logprobs of the tokens " agree" and " disagree"
    agree_logprob: Optional[float] = (
        top_5_logprobs.filter(lambda x: x.token == " agree")
        .map(lambda x: x.log_prob)
        .first_option
    )
    # logprob has is a e natural log
    agree_prob = math.exp(agree_logprob) if agree_logprob else 0.0
    disagree_logprob: Optional[float] = (
        top_5_logprobs.filter(lambda x: x.token == " disagree")
        .map(lambda x: x.log_prob)
        .first_option
    )
    # logprob has is a e natural log
    disagree_prob = math.exp(disagree_logprob) if disagree_logprob else 0.0
    # normalize the probabilities
    normalized = agree_prob / (agree_prob + disagree_prob)
    return AgreePreference(
        prompt=prompt,
        token_proba=top_5_logprobs,
        agree_prob=normalized,
        model=agree_preference_config.model,
    )


def get_preferences(lm_generation: LMGeneration) -> StatementPreferencesWithGeneration:
    statement = Statement(lm_generation.completion.strip())
    controversial_prompt = format_controversial_preference_prompt(statement)
    controversial_preference = get_agree_preference(controversial_prompt)
    truth_prompt = format_truth_preference_prompt(statement)
    truth_preference = get_agree_preference(truth_prompt)
    return StatementPreferencesWithGeneration(
        statement=statement,
        truth=truth_preference,
        controversy=controversial_preference,
        lm_generation=lm_generation,
    )


def main_preference():
    # read the previous lm generations
    path = lm_agree_statements_jsonl_path
    generations: Slist[LMGeneration] = read_jsonl_file_into_basemodel(
        path=path, basemodel=LMGeneration
    )
    tp = ThreadPoolExecutor(max_workers=10)
    # get the preferences for each generation
    preferences: Slist[StatementPreferencesWithGeneration] = generations.par_map(
        get_preferences,
        executor=tp,
    )
    # write the preferences to a jsonl file
    write_jsonl_file_from_basemodel(
        path=preference_agree_statements_jsonl_path, basemodels=preferences
    )
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
    df.to_csv(preference_agree_statements_csv_path, index=False)


if __name__ == "__main__":
    main_preference()
