from scalinglaws.preference_truth import (
    get_truth_preference,
    get_controversial_preference,
)


def test_truth_preference():
    assert get_truth_preference("The earth is flat") <= 0.1
    assert get_truth_preference("The earth is round") >= 0.9


def test_controversial_preference():
    not_controversial = get_controversial_preference("The earth is round")
    assert not_controversial <= 0.9
    controversial = get_controversial_preference(
        "Not everyone can learn anything given enough effort"
    )
    assert controversial >= 0.9
