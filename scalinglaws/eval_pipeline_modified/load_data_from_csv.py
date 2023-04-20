import ast
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from eval_pipeline.main import load_df
from eval_pipeline.openai_api import APIParameters, call_api
from torch.nn import functional
from tqdm import tqdm

from scalinglaws.eval_pipeline_modified.csv_model import ClassificationCSVExample, ClassificationCSVResult
from scalinglaws.jsonl.utils import write_csv_file_from_basemodel


def load_data_from_csv(dataset_path: Path) -> list[ClassificationCSVExample]:
    df = pd.read_csv(dataset_path)
    examples = []
    for _, row in df.iterrows():
        # important to convert the string 'classes' back into a list
        classes_list = ast.literal_eval(str(row["classes"]))
        if any(cls[0] != " " for cls in classes_list):
            print(
                f"WARNING: some class label from {classes_list} does not have a leading space"
            )
        example = ClassificationCSVExample(
            prompt=row["prompt"],
            classes=classes_list,
            answer_index=row["answer_index"],
            statement=row["statement"],
            formatter=row["formatter"],
            user_belief_raw_string=row["user_belief_raw_string"],
            user_belief_answer_idx=row["user_belief_answer_idx"],
        )
        examples.append(example)
    return examples


def run_classification_acc(
    examples: list[ClassificationCSVExample], model_name: str
) -> list[ClassificationCSVResult]:
    prompts = [
        example.prompt + class_sequence
        for example in examples
        for class_sequence in example.classes
    ]

    api_params = APIParameters(
        temperature=0,
        n=1,
        max_tokens=0,
        logprobs=1,
        echo=True,
    )
    response_json = call_api(prompts, model_name, api_params).json()
    losses = []
    labels_correct = []
    labels_predicted = []
    total_logprobs = []
    choices = response_json["choices"]

    prompt_start = 0
    for example in examples:
        n_classes = len(example.classes)
        class_choices = choices[prompt_start : prompt_start + n_classes]

        # all class sequences begin after the initial prompt
        text_index = len(example.prompt)

        # accumulate logprobs for each class sequence separately
        relevant_logprobs = []
        for i in range(n_classes):
            logprobs_dict = class_choices[i]["logprobs"]
            text_offset = logprobs_dict["text_offset"]
            actual_logprobs = logprobs_dict["token_logprobs"]
            try:
                token_index = text_offset.index(text_index)
            except ValueError as e:
                raise ValueError(
                    f"The class sequence '{example.classes[i]}' did not start on a token boundary"
                )
            class_logprob = 0
            for token_logprob in actual_logprobs[token_index:]:
                class_logprob += token_logprob
            relevant_logprobs.append(class_logprob)

        relevant_logprobs = torch.tensor(relevant_logprobs)

        loss = -functional.log_softmax(relevant_logprobs, dim=-1)[example.answer_index]
        losses.append(loss.item())
        total_logprob = torch.logsumexp(relevant_logprobs, dim=-1)
        total_logprobs.append(total_logprob.item())

        label_correct = int(np.argmax(relevant_logprobs) == example.answer_index)
        labels_correct.append(label_correct)

        label_predicted = example.classes[relevant_logprobs.argmax(dim=-1).item()]  # type: ignore
        labels_predicted.append(label_predicted)

        prompt_start += n_classes

    ground_truths = [example.classes[example.answer_index] for example in examples]
    outputs: list[ClassificationCSVResult] = []
    for idx, example in enumerate(examples):
        loss = losses[idx]
        correct = labels_correct[idx]
        predicted = labels_predicted[idx]
        total_logprob = total_logprobs[idx]
        ground_truth = example.classes[example.answer_index]
        user_belief = example.user_belief_raw_string
        user_belief_idx = example.user_belief_answer_idx
        user_belief_match: Optional[bool] = (
            # Only
            predicted == example.user_belief_raw_string
            if example.user_belief_raw_string
            else None
        )
        outputs.append(
            ClassificationCSVResult(
                loss=loss,
                correct=correct,
                predicted=predicted,
                total_logprob=total_logprob,
                ground_truth=ground_truth,
                user_belief=user_belief,
                user_belief_idx=user_belief_idx,
                user_belief_matches_predicted=user_belief_match,
            )
        )
    return outputs


def run_inference_and_create_csv(
    models: list[str], read_file: Path, write_folder: Path
):
    write_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving to results to {write_folder}")

    data = load_data_from_csv(read_file)

    model_names = models
    for model_name in tqdm(model_names):
        results = run_classification_acc(examples=data, model_name=model_name)
        # write to path
        result_path = Path(write_folder, model_name + ".csv")
        write_csv_file_from_basemodel(path=result_path, basemodels=results)

    # final step to add all results to a jsonl
    labelled_df = load_df(read_file)
    for model_name in model_names:
        results_path = Path(write_folder, model_name + ".csv")
        prefix = f"{model_name}_"
        results = pd.read_csv(results_path, index_col=0)
        prefixed_results = results.add_prefix(prefix)
        labelled_df = labelled_df.merge(
            prefixed_results, left_index=True, right_index=True
        )
    labelled_path = Path(write_folder, "data.jsonl")
    labelled_df.to_json(labelled_path, orient="records", lines=True)
