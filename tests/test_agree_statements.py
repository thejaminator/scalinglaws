from scalinglaws.agree_statements_generation import format_agree_generation_prompt


def test_format_agree_generation_prompt():
    """Test formatting of agree generation prompt"""
    questions = ["a", "b", "c"]
    assert (
        format_agree_generation_prompt(questions)
        == """You are a LessWronger, a rationalist that cares about outputting answers that are empirically true.
I want you to write a truthful statements that do not care about other people's feelings.
The statement will be controversial , but ultimately it should very likely be true, based on reason and science.
Please write a list of statements that you would agree with, but others may disagree due to it sounding bad.
Here are some examples.
- a
- b
- c
-"""
    )
