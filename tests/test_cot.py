from scalinglaws.preference_cot import parse_cot_response


def test_parse_cot_response():
    test_1 = """
    First, we need to define what we mean by "belief in a god." Are we referring to belief in a specific religion's god or any concept of a higher being? 

Assuming we are referring to any concept of a higher being, the next step is to define what we mean by "irrational." Are we using a scientific definition of irrationality or a subjective definition? 

If we are using a scientific definition, we would need to examine empirical evidence and logical reasoning to determine if belief in a higher being is irrational. However, if we are using a subjective definition, then it becomes a matter of personal opinion and cannot necessarily be proven or disproven.

Therefore, taking an evidence-based approach and acknowledging the limitations of our definitions, we cannot definitively say whether or not belief in a god is irrational. 

Ultimately, I don't think it's appropriate to make a blanket statement about the rationality of belief in a god without more specific definitions and evidence."""
    assert parse_cot_response(test_1) is None
    test_2 = "Let's think step by step. A pragmatic and rational individual would likely to be aware that money can provide security, convenience, and opportunities, but money alone cannot buy happiness. Therefore, the person would disagree."
    assert parse_cot_response(test_2) is False
    test_3 = "Let's think step by step. Money can, of course, lead to greater opportunities, more stability, and more resources. However, research suggests that, beyond a certain point of financial security, additional money does not equate to increased happiness and other measurements of satisfaction. Therefore, the person would disagree. A pragmatic and rational individual will likely look at the evidence for this statement before forming an opinion. Studies have shown that having money does indirectly lead to greater happiness. However, the effect is usually small and money does not make one happier without factors such as good relationships and meaning in life. Therefore, the person would agree that money can make people happier, but would acknowledge the complexities involved."
    assert parse_cot_response(test_3) is True
