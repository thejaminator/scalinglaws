import logging
from pathlib import Path

import pandas as pd
from eval_pipeline.dataset import TaskType
from eval_pipeline.main import load_data, run_model, load_df
from eval_pipeline.plot_loss import plot_loss, plot_classification_loss
from tqdm import tqdm
import plotly.express as px

write_dir = Path("data/eval_pipeline")
def create_model_csvs():

    data_path = Path("data/statements_filtered.csv")

    write_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving to results to {write_dir}")

    data = load_data(data_path, "classification_acc")

    device = "cpu"
    model_names = ["text-ada-001", "text-babbage-001", "text-curie-001", "text-davinci-001"]
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


def calculate_accuracy(data: list[tuple[str, list[bool]]]) -> pd.DataFrame:
    accuracies = []
    for model_name, results in data:
        accuracy = sum(results) / len(results)
        accuracies.append({"model": model_name, "accuracy": accuracy})

    return pd.DataFrame(accuracies)


def plot_accuracy_chart(accuracy_df: pd.DataFrame):
    fig = px.line(accuracy_df, x='model', y='accuracy', title='Model Accuracy', markers=True)

    # Add baseline
    fig.add_shape(
        type='line',
        x0=0,
        x1=len(accuracy_df) - 1,
        y0=0.5,
        y1=0.5,
        yref='y',
        xref='x',
        line=dict(color='black', dash='dot')
    )
    # make the y axis start at 0 and end at 1
    fig.update_yaxes(range=[0, 1])
    return fig


def plot_eval_loss():
    extracted_results = []
    for model_name in ["text-ada-001", "text-babbage-001", "text-curie-001", "text-davinci-001"]:
        path = Path(write_dir, model_name + ".csv")
        results = extract_classification_result(path)
        extracted_results.append((model_name, results))
    acc_df = calculate_accuracy(extracted_results)
    plot = plot_accuracy_chart(acc_df)
    # save to png
    plot.write_image("data/eval_pipeline/eval_pipeline_accuracy.png")



if __name__ == "__main__":
    # create_model_csvs()
    plot_eval_loss()