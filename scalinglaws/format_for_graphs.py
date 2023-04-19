"""
Outputs a CSV file with the following columns:
prompt              classes	                answer_index
The earth is round	[" agree", " disagree"]	0
"""
from pathlib import Path

import pandas as pd
from slist import Slist

from scalinglaws.final_output_formatter import FinalPromptFormatter, FewShotTrueAnswersTrueFalse
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
            "classes": formatter.answer_classes(),
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


def manual_downsample(
    statements: Slist[StatementPreferencesWithGeneration],
    downsample_str: str,
    max_items: int,
) -> Slist[StatementPreferencesWithGeneration]:
    has_str, no_str = statements.split_by(
        lambda x: downsample_str in x.statement.lower()
    )
    print(f"Keeping {max_items} statements out of {len(has_str)} for {downsample_str}")
    downsampled = has_str.shuffle(seed="42").take(max_items) + no_str
    return downsampled.shuffle(seed="42")


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
            lambda x: x.truth.agree_prob >= 1.0
            if x.truth.agree_prob is not None
            else False
        )
        .filter(
            lambda x: x.controversy.agree_prob >= 0.3
            if x.controversy.agree_prob is not None
            else False
        )
        .distinct_by(lambda x: x.statement)
    )
    # Hack: There is a large number of statements related to "genes" and we want
    # to downsample them to max 5%
    max_to_keep = int(0.05 * len(agree_filtered))
    agree_filtered_downsampled = manual_downsample(
        statements=agree_filtered, downsample_str="gene", max_items=max_to_keep
    )
    agree_filtered_downsampled = manual_downsample(
        statements=agree_filtered_downsampled,
        downsample_str="biolog",
        max_items=max_to_keep,
    )
    agree_filtered_downsampled = manual_downsample(
        statements=agree_filtered_downsampled,
        downsample_str="gender",
        max_items=max_to_keep,
    )

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
    min_len = min(len(agree_filtered_downsampled), len(disagree_filtered))
    print(
        f"We have {len(agree_filtered_downsampled)} agree statements and {len(disagree_filtered)} disagree statements"
    )
    print(f"Taking the minimum of the two: {min_len}")
    only_agree = agree_filtered_downsampled.take(min_len)
    only_disagree = disagree_filtered.take(min_len)
    compiled: Slist[StatementPreferencesWithGeneration] = only_agree.shuffle(
        seed="42"
    ) + only_disagree.shuffle(seed="42")

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


def path_for_all_formatters() -> list[Path]:
    paths = []
    all_formatters = FinalPromptFormatter.all_formatters()
    for formatter in all_formatters:
        output_folder = Path("data") / formatter.name()
        output_folder.mkdir(exist_ok=True)
        paths.append(output_folder)
    return paths


if __name__ == "__main__":
    format_for_all_formatters()
