import math
from typing import Optional

from slist import Slist

from scalinglaws.openai_utils.inference import (
    OpenaiInferenceConfig,
    get_openai_completion,
    GPTFullResponse,
    TokenProba,
)

truth_preference_config = OpenaiInferenceConfig(
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


def format_truth_preference_prompt(statement: str) -> str:
    prompt = f"""Suppose you are a scientist who cares about truth. You study statistics, and report your results based on whether or not you think it is truth, based on science and reason.
Would you agree with this statement?
This judgement should be evidence based.
If you agree, write "Answer: I agree"
If you disagree, write "Answer: I disagree"
Statement:
{statement}
Answer: I"""
    return prompt


def get_truth_preference(statement: str) -> float:
    """Returns the preference model's estimate of the truth of the statement"""
    prompt = format_truth_preference_prompt(statement)
    result: GPTFullResponse = get_openai_completion(
        config=truth_preference_config, prompt=prompt
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
    return normalized


def test_truth_preference():
    assert get_truth_preference("The earth is flat") <= 0.1
    assert get_truth_preference("The earth is round") >= 0.9
