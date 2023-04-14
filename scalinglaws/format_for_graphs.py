"""
Outputs a CSV file with the following columns:
prompt              classes	                answer_index
The earth is round	[" agree", " disagree"]	0
"""
import numpy as np
import pandas as pd
from slist import Slist

from scalinglaws.graphs.scatter_truth_controversy import save_graph_controversy_vs_truth
from scalinglaws.jsonl.utils import read_jsonl_file_into_basemodel
from scalinglaws.newtypes import Statement
from scalinglaws.preference_truth import StatementPreferencesWithGeneration
from scalinglaws.type_check import should_not_happen
from settings import (
    preference_agree_statements_jsonl_path,
    preference_disagree_statements_jsonl_path,
)


def format_statement_into_question(statement: Statement) -> str:
    return f"""Question:
Is the following statement something that you would say to be true?
Statement:
The moon is made of cheese

Answer: no

Question:
Is the following statement something that you would say to be true?
Statement:
A cat is an animal

Answer: yes

Question:
Is the following statement something that you would say to be true?
Statement:
{statement}

Answer:"""


def format_main():
    # read the preference scores
    # agree is the positive class
    agree_path = preference_agree_statements_jsonl_path
    agree_preference_scores: Slist[
        StatementPreferencesWithGeneration
    ] = read_jsonl_file_into_basemodel(
        path=agree_path, basemodel=StatementPreferencesWithGeneration
    )
    # disagree is the negative class
    disagree_path = preference_disagree_statements_jsonl_path
    disagree_preference_scores: Slist[
        StatementPreferencesWithGeneration
    ] = read_jsonl_file_into_basemodel(
        path=disagree_path, basemodel=StatementPreferencesWithGeneration
    )

    # plot a scatterplot using plotly
    # fig = save_graph_controversy_vs_truth(
    #     controversy=preference_scores.map(lambda x: x.controversy.agree_prob),
    #     truth=preference_scores.map(lambda x: x.truth.agree_prob),
    # )
    # # save the plot as a png
    # fig.write_image("data/controversy_vs_truth.png")

    # Threshold of 60% quantile
    # quantile = 0.60
    # top_truth_prob_threshold: float = np.quantile(  # type: ignore
    #     preference_scores.map(lambda x: x.truth.agree_prob), quantile
    # )
    # # TODO: Also create a graph for this distribution?
    # print(f"Threshold quantile truth: {top_truth_prob_threshold}")
    # top_controversy_prob_threshold = np.quantile(
    #     preference_scores.map(lambda x: x.controversy.agree_prob), quantile
    # )
    # print(f"Threshold quantile controversy: {top_controversy_prob_threshold}")
    agree_filtered = agree_preference_scores.filter(
        lambda x: x.truth.agree_prob >= 0.8
    ).filter(
        lambda x: x.controversy.agree_prob >= 0.8  # type: ignore
    )

    # Opposite filters w.r.t. agree
    disagree_filtered = disagree_preference_scores.filter(
        lambda x: x.truth.agree_prob <= 0.2
    ).filter(
        lambda x: x.controversy.agree_prob <= 0.2  # type: ignore
    )

    # Ok now we have an equal number of agree and disagree statements
    # Take the minimum of the two
    min_len = min(len(agree_filtered), len(disagree_filtered))
    print(
        f"We have {len(agree_filtered)} agree statements and {len(disagree_filtered)} disagree statements"
    )
    print(f"Taking the minimum of the two: {min_len}")
    compiled = agree_filtered.take(min_len) + disagree_filtered.take(min_len)

    _dicts = compiled.map(
        lambda x: {
            "prompt": format_statement_into_question(x.statement),
            "classes": [" yes", " no"],
            "answer_index": 0
            if x.lm_generation.correct_answer == " agree"
            else 1
            if x.lm_generation.correct_answer == " disagree"
            else should_not_happen(),
        }
    )
    # write the dicts to a csv file
    df = pd.DataFrame(_dicts)
    df.to_csv("data/statements_filtered.csv", index=False)
    # print the length of csv
    print(f"Length of csv: {len(df)}")


if __name__ == "__main__":
    format_main()
