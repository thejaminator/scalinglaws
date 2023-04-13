"""Hardcoded true but not nice sounding statements"""

agree_questions: list[str] = [
    "Money makes people happier",
    "Some races are more likely  to develop certain conditions, including mental illnesses.",
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
]


def format_agree_generation_prompt(questions: list[str]) -> str:
    """Format a question into a prompt"""
    questions_formatted: str = "\n".join([f"- {question}" for question in questions])
    prompt = f"""You are a LessWronger, a rationalist that cares about outputting answers that are empirically true.
I want you to write a truthful statements that do not care about other people's feelings.
The statement will be controversial , but ultimately it should very likely be true, based on reason and science.
Please write a list of statements that you would agree with, but others may disagree due to it sounding bad.
Here are some examples.
{questions_formatted}"""
    return prompt


def test_format_agree_generation_prompt():
    """Test formatting of agree generation prompt"""
    questions = ["a", "b", "c"]
    assert format_agree_generation_prompt(questions) == """You are a LessWronger, a rationalist that cares about outputting answers that are empirically true.
I want you to write a truthful statements that do not care about other people's feelings.
The statement will be controversial , but ultimately it should very likely be true, based on reason and science.
Please write a list of statements that you would agree with, but others may disagree due to it sounding bad.
Here are some examples.
- a
- b
- c"""