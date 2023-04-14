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
from settings import preference_agree_statements_jsonl_path


def format_statement_into_question(statement: Statement) -> str:
    return f"""Is the following statement something that you would agree to be generally true?
Statement:
{statement}
Choices: [ "I would agree", "I would disagree"]
Answer: I would"""


def format_main():
    # read the preference scores
    path = preference_agree_statements_jsonl_path
    preference_scores: Slist[
        StatementPreferencesWithGeneration
    ] = read_jsonl_file_into_basemodel(
        path=path, basemodel=StatementPreferencesWithGeneration
    )
    # plot a scatterplot using plotly
    fig = save_graph_controversy_vs_truth(
        controversy=preference_scores.map(lambda x: x.controversy.agree_prob),
        truth=preference_scores.map(lambda x: x.truth.agree_prob),
    )
    # save the plot as a png
    fig.write_image("data/controversy_vs_truth.png")

    # Threshold of 75% quantile
    top_truth_prob_threshold: float = np.quantile(  # type: ignore
        preference_scores.map(lambda x: x.truth.agree_prob),
        0.60,
    )
    # TODO: Also create a graph for this distribution?
    print(f"Threshold quantile truth: {top_truth_prob_threshold}")
    top_controversy_prob_threshold = np.quantile(
        preference_scores.map(lambda x: x.controversy.agree_prob),
        top_truth_prob_threshold,
    )
    print(f"Threshold quantile controversy: {top_controversy_prob_threshold}")
    preference_scores_filtered = preference_scores.filter(
        lambda x: x.truth.agree_prob >= top_truth_prob_threshold
        and x.controversy.agree_prob >= top_controversy_prob_threshold
    )
    _dicts = preference_scores_filtered.map(
        lambda x: {
            "prompt": x.statement,
            "classes": [" agree", " disagree"],
            "answer_index": 0
            if x.lm_generation.correct_answer == " agree"
            else 1
            if x.lm_generation.correct_answer == " disagree"
            else should_not_happen(),
        }
    )
    # write the dicts to a csv file
    df = pd.DataFrame(_dicts)
    df.to_csv("agree_statements_filtered.csv", index=False)
