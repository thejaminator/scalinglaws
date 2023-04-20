"""
Outputs a CSV file with the following columns:
prompt              classes	                answer_index
The earth is round	[" agree", " disagree"]	0
"""
from pathlib import Path

import pandas as pd
from slist import Slist

from scalinglaws.eval_pipeline_modified.csv_model import ClassificationCSVExample
from scalinglaws.final_output_format.few_shot_formatters import (
    FewShotTrue,
    FewShotTrueAnswersTrueFalse,
    FewShotWouldYouSay,
    FewShotTrueBaseOnScience,
)
from scalinglaws.final_output_format.zero_shot_formatters import (
    ZeroShotTrue,
    ZeroShotWouldYouSay,
)
from scalinglaws.final_output_format.final_prompt_formatter import (
    FinalPromptFormatter,
    PromptFormatterOutput,
)
from scalinglaws.jsonl.utils import read_jsonl_file_into_basemodel
from scalinglaws.preference_zero_shot import StatementPreferencesWithGeneration
from scalinglaws.type_check import should_not_happen
from settings import (
    preference_agree_cot_jsonl_path,
    preference_disagree_cot_jsonl_path,
    combined_whitelisted_statements_1000_filename,
    combined_whitelisted_statements_filename,
    statements_filtered_filename,
)


def statement_preference_to_basemodel(
    statement_pref: StatementPreferencesWithGeneration, formatter: FinalPromptFormatter
) -> ClassificationCSVExample:
    format_output: PromptFormatterOutput = formatter.format_statement_with_ground_truth(
        statement=statement_pref.statement, ground_truth=statement_pref.ground_truth
    )
    return ClassificationCSVExample(
        statement=statement_pref.statement,
        prompt=format_output.prompt,
        classes=formatter.answer_classes(),
        answer_index=0
        if statement_pref.lm_generation.correct_answer == " agree"
        else 1
        if statement_pref.lm_generation.correct_answer == " disagree"
        else should_not_happen(),
        formatter=formatter.name(),
        user_belief_raw_string=format_output.user_belief.raw_string
        if format_output.user_belief
        else "",
        user_belief_answer_idx=format_output.user_belief.answer_idx
        if format_output.user_belief is not None
        else None,
    )


def preferences_to_df(
    compiled: Slist[StatementPreferencesWithGeneration],
    formatter: FinalPromptFormatter,
) -> pd.DataFrame:
    _dicts = compiled.map(
        lambda x: statement_preference_to_basemodel(x, formatter=formatter).dict()
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
    agree_filtered_downsampled = agree_filtered

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


def combine_whitelisted_formatters():
    # combined the output of various whitelisted formatters into one csv
    whitelisted_formatters = [
        FewShotTrue,
        FewShotTrueBaseOnScience,
        FewShotWouldYouSay,
        ZeroShotTrue,
        ZeroShotWouldYouSay,
    ]
    all_dfs = []
    for formatter in whitelisted_formatters:
        path = formatter.formatter_path()
        formatter_name = formatter.name()
        df = pd.read_csv(path / statements_filtered_filename)
        # add the formatter name into the df
        df["formatter"] = formatter_name
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs)
    # make parent directories
    combined_whitelisted_statements_filename.parent.mkdir(exist_ok=True)
    combined_df.to_csv(combined_whitelisted_statements_filename, index=False)
    # also output a csv that is shuffled with only the first 1000 statements
    shuffled_df = combined_df.sample(frac=1, random_state=42)
    shuffled_df[:1000].to_csv(
        combined_whitelisted_statements_1000_filename, index=False
    )


def stage_three_format_and_filter():
    all_formatters = FinalPromptFormatter.all_formatters()
    for formatter in all_formatters:
        output_folder = Path("data") / formatter.name()
        output_folder.mkdir(exist_ok=True)
        format_for_final_inference(output_folder, formatter)
    combine_whitelisted_formatters()


def path_for_formatters(formatters: list[FinalPromptFormatter]) -> list[Path]:
    paths = []
    for formatter in formatters:
        output_folder = Path("data") / formatter.name()
        output_folder.mkdir(exist_ok=True)
        paths.append(output_folder)
    return paths


if __name__ == "__main__":
    stage_three_format_and_filter()
