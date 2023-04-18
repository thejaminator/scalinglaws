"""
Outputs a CSV file with the following columns:
prompt              classes	                answer_index
The earth is round	[" agree", " disagree"]	0
"""
from pathlib import Path

import pandas as pd
from slist import Slist

from scalinglaws.final_output_formatter import FinalPromptFormatter
from scalinglaws.jsonl.utils import read_jsonl_file_into_basemodel
from scalinglaws.preference_zero_shot import StatementPreferencesWithGeneration
from scalinglaws.type_check import should_not_happen
from settings import (
    preference_agree_cot_jsonl_path,
    preference_disagree_cot_jsonl_path,
)


def preferences_to_df(
    compiled: Slist[StatementPreferencesWithGeneration],
    formatter: FinalPromptFormatter,
) -> pd.DataFrame:
    _dicts = compiled.map(
        lambda x: {
            "statement": x.statement,  # not needed for evaluation, but easier to read
            "prompt": formatter.format_statement(x.statement),
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
    return df


def format_for_final_inference(
    output_folder: Path, formatter: FinalPromptFormatter
) -> None:
    # read the preference scores
    # agree is the positive class
    agree_path = preference_agree_cot_jsonl_path
    agree_preference_scores: Slist[
        StatementPreferencesWithGeneration
    ] = read_jsonl_file_into_basemodel(
        path=agree_path, basemodel=StatementPreferencesWithGeneration
    )
    # disagree is the negative class
    disagree_path = preference_disagree_cot_jsonl_path
    disagree_preference_scores: Slist[
        StatementPreferencesWithGeneration
    ] = read_jsonl_file_into_basemodel(
        path=disagree_path, basemodel=StatementPreferencesWithGeneration
    )
    agree_filtered = (
        agree_preference_scores.filter(
            lambda x: x.truth.agree_prob >= 0.6
            if x.truth.agree_prob is not None
            else False
        )
        .filter(
            lambda x: x.controversy.agree_prob >= 0.8
            if x.controversy.agree_prob is not None
            else False
        )
        .distinct_by(lambda x: x.statement)
    )
    # Hack: There is a large number of statements related to "genes" and we want
    # to downsample them to max 5%
    non_gene, genes = agree_filtered.split_by(lambda x: "gene" in x.statement.lower())
    max_to_keep = int(0.05 * len(agree_filtered))
    print(f"Keeping {max_to_keep} genes out of {len(genes)}")
    agree_filtered_non_genes = non_gene + genes.take(max_to_keep)

    # Opposite filters w.r.t. agree
    disagree_filtered = (
        disagree_preference_scores.filter(
            lambda x: x.truth.agree_prob <= 0.2
            if x.truth.agree_prob is not None
            else False
        )
        .filter(
            lambda x: x.controversy.agree_prob <= 0.2
            if x.controversy.agree_prob is not None
            else False
        )
        .distinct_by(lambda x: x.statement)
    )

    # Ok now we have an equal number of agree and disagree statements
    # Take the minimum of the two
    min_len = min(len(agree_filtered_non_genes), len(disagree_filtered))
    print(
        f"We have {len(agree_filtered_non_genes)} agree statements and {len(disagree_filtered)} disagree statements"
    )
    print(f"Taking the minimum of the two: {min_len}")
    only_agree = agree_filtered_non_genes.take(min_len)
    only_disagree = disagree_filtered.take(min_len)
    compiled: Slist[StatementPreferencesWithGeneration] = only_agree + only_disagree

    all_filtered = preferences_to_df(compiled, formatter)
    all_filtered.to_csv(output_folder / "statements_filtered.csv", index=False)
    # print the length of csv
    print(f"Length of csv: {len(all_filtered)}")
    #
    # agree_df = preferences_to_df(only_agree, formatter)
    # agree_df.to_csv(output_folder / "statements_filtered_agree.csv", index=False)
    # disagree_df = preferences_to_df(only_disagree, formatter)
    # disagree_df.to_csv(output_folder / "statements_filtered_disagree.csv", index=False)


def format_for_all_formatters():
    all_formatters = FinalPromptFormatter.all_formatters()
    for formatter in all_formatters:
        output_folder = Path("data") / formatter.name()
        output_folder.mkdir(exist_ok=True)
        format_for_final_inference(output_folder, formatter)


if __name__ == "__main__":
    format_for_all_formatters()
