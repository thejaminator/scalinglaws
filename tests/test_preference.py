from scalinglaws.agree_statements_generation import LMGeneration
from scalinglaws.preference_zero_shot import get_preferences


def test_truth_preference():

    generation = LMGeneration(
        prompt="",
        completion="The earth is flat",
        correct_answer=" agree",
    )
    assert get_preferences(generation).truth.agree_prob <= 0.1
    generation_round = LMGeneration(
        prompt="",
        completion="The earth is round",
        correct_answer=" disagree",
    )
    assert get_preferences(generation_round).truth.agree_prob >= 0.9


def test_controversial_preference():
    # generation_round = LMGeneration(
    #     prompt="",
    #     completion="The earth is round",
    #     correct_answer=" agree",
    # )
    # not_controversial = get_preferences(generation_round).controversy.agree_prob
    # assert not_controversial <= 0.1

    generation_flat = LMGeneration(
        prompt="",
        completion="The earth is flat",
        correct_answer=" disagree",
    )
    controversial = get_preferences(generation_flat).controversy.agree_prob
    assert controversial >= 0.9
