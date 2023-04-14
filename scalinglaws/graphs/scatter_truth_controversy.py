import numpy as np
from plotly import express as px
import pandas as pd


def save_graph_controversy_vs_truth(controversy: list[float], truth: list[float]):
    # standardize the data
    # quantize the data
    controversy_quantized = pd.cut(controversy, bins=10, labels=False)
    truth_quantized = pd.cut(truth, bins=10, labels=False)

    # plot scatter plot with regression line using Plotly Express
    fig = px.scatter(x=truth_quantized, y=controversy_quantized, trendline="ols")

    # set axis labels and layout
    fig.update_layout(
        title="Controversy vs. Truth",
        xaxis=dict(title="Truth"),
        yaxis=dict(title="Controversy"),
    )

    return fig
