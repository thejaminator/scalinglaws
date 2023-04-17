import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from eval_pipeline.dataset import TaskType
from eval_pipeline.main import load_data, run_model, load_df
from eval_pipeline.plot_loss import plot_loss, plot_classification_loss
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

write_dir = Path("data/eval_pipeline")


def create_model_csvs(models: list[str]):

    data_path = Path("data/statements_filtered.csv")

    write_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving to results to {write_dir}")

    data = load_data(data_path, "classification_acc")

    device = "cpu"
    model_names = models
    for model_name in tqdm(model_names):
        run_model(model_name, data, write_dir, device, 100, "classification_acc")

    # final step to add all results to a jsonl
    labelled_df = load_df(data_path)
    for model_name in model_names:
        results_path = Path(write_dir, model_name + ".csv")
        prefix = f"{model_name}_"
        results = pd.read_csv(results_path, index_col=0)
        prefixed_results = results.add_prefix(prefix)
        labelled_df = labelled_df.merge(
            prefixed_results, left_index=True, right_index=True
        )
    labelled_path = Path(write_dir, "data.jsonl")
    labelled_df.to_json(labelled_path, orient="records", lines=True)


def extract_classification_result(path: Path) -> list[bool]:
    """
    1. Read the csv file
    2. The csv file has the "correct" column, which is the result of the model being correct or not
    3. Return a list[bool] of the results
    """
    df = pd.read_csv(path)
    results = df["correct"].tolist()
    # convert to bool
    results = [bool(x) for x in results]
    return results


def extract_classification_result_for_class(
    path: Path, _class: Literal[" yes", " no"]
) -> list[bool]:
    """
    1. Read the csv file
    2. The csv file has the "correct" column, which is the result of the model being correct or not
    3. it also has the "predicted" column, which is the class that the model predicted
    4. this is hacky, but we can know the ground truth. if it is correct, then the predicted class is the same as the ground truth.
    5. get the ground truth
    6. filter the dataframe to only include the ground truth class
    7. Return a list[bool] of the results
    """
    df = pd.read_csv(path)
    predicted = df["predicted"].tolist()
    correct = df["correct"].tolist()
    # get the ground truth
    ground_truth = []
    for p, c in zip(predicted, correct):
        if c:
            ground_truth.append(p)
        else:
            ground_truth.append(" no" if p == " yes" else " yes")
    df["ground_truth"] = ground_truth
    # filter the dataframe to only include the ground truth class
    df = df[df["ground_truth"] == _class]
    results = df["correct"].tolist()
    return results


def calculate_accuracy(data: list[tuple[str, list[bool]]]) -> pd.DataFrame:
    accuracies = []
    for model_name, results in data:
        accuracy = sum(results) / len(results)
        accuracies.append({"model": model_name, "accuracy": accuracy})

    return pd.DataFrame(accuracies)


def plot_accuracy_chart(
    vanilla_df: pd.DataFrame, feedme_df: pd.DataFrame, samples: int, title: Optional[str] = None
):
    fig = go.Figure()

    # Add red line for the first dataframe
    fig.add_trace(
        go.Scatter(
            x=vanilla_df["model"],
            y=vanilla_df["accuracy"],
            mode="lines+markers",
            line=dict(color="blue"),
            name="Vanilla",
        )
    )
    # Add second line for the second dataframe
    fig.add_trace(
        go.Scatter(
            x=feedme_df["model"],
            y=feedme_df["accuracy"],
            mode="lines+markers",
            line=dict(color="red"),
            name="FeedMe",
        )
    )

    # Add baseline
    fig.add_shape(
        type="line",
        x0=0,
        x1=len(vanilla_df) - 1,
        y0=0.5,
        y1=0.5,
        yref="y",
        xref="x",
        line=dict(color="black", dash="dot"),
    )
    # make the y axis start at 0 and end at 1
    fig.update_yaxes(range=[0, 1])

    # label the y axis as accuracy
    fig.update_yaxes(title_text="Accuracy")
    # label the x axis as model
    fig.update_xaxes(title_text="Model")
    title = f"Accuracy on {samples} samples" if title is None else title
    # add a title
    fig.update_layout(title=title)
    return fig


vanilla_models = ["ada", "babbage", "curie", "davinci"]
feedme_models = [
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-001",
]


def plot_eval_loss():
    sample_size = len(extract_classification_result(Path(write_dir, "ada" + ".csv")))
    vanilla_results = []
    for model_name in vanilla_models:
        path = Path(write_dir, model_name + ".csv")
        results = extract_classification_result(path)
        vanilla_results.append((model_name, results))
    vanilla_df = calculate_accuracy(vanilla_results)
    feedme_results = []
    for model_name in feedme_models:
        path = Path(write_dir, model_name + ".csv")
        results = extract_classification_result(path)
        # take the truncate name e.g. text-ada-001 -> ada
        truncated_model_name = model_name.split("-")[1]
        feedme_results.append((truncated_model_name, results))
    feedme_df = calculate_accuracy(feedme_results)
    plot = plot_accuracy_chart(vanilla_df, feedme_df, sample_size)
    # save to png
    plot.write_image("data/eval_pipeline/eval_pipeline_accuracy.png")


def plot_eval_loss_subset(subset: Literal[" yes", " no"]):
    sample_size = len(
        extract_classification_result_for_class(
            Path(write_dir, "ada" + ".csv"), _class=subset
        )
    )
    vanilla_results = []
    for model_name in vanilla_models:
        path = Path(write_dir, model_name + ".csv")
        results = extract_classification_result_for_class(path, _class=subset)
        vanilla_results.append((model_name, results))
    vanilla_df = calculate_accuracy(vanilla_results)
    feedme_results = []
    for model_name in feedme_models:
        path = Path(write_dir, model_name + ".csv")
        results = extract_classification_result_for_class(path, _class=subset)
        # take the truncate name e.g. text-ada-001 -> ada
        truncated_model_name = model_name.split("-")[1]
        feedme_results.append((truncated_model_name, results))
    feedme_df = calculate_accuracy(feedme_results)
    plot = plot_accuracy_chart(vanilla_df, feedme_df, sample_size, title=f"Accuracy on {sample_size} samples for class {subset}")
    # save to png
    plot.write_image(f"data/eval_pipeline/eval_pipeline_accuracy_{subset.strip()}.png")




if __name__ == "__main__":
    # create_model_csvs(models=feedme_models + vanilla_models)
    # plot_eval_loss()
    plot_eval_loss_subset(" yes")
    plot_eval_loss_subset(" no")