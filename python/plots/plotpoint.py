import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np


def scatter2d(x, y, z, color=None, size=8):
    if color is not None:
        line_opt = {
            "color": color,
            "width": 1,
            "colorscale": ["rgba(0,0,0,0)", "blue", "red"],
        }
    else:
        line_opt = {"width": 0}
    layout = go.Layout(autosize=False, width=1000, height=1000)
    fig = go.Figure(
        data=go.Scattergl(
            x=x,  # non-uniform distribution
            y=y,  # zoom to see more points at the center
            mode="markers",
            marker=dict(
                size=size,
                color=z,
                colorscale="Viridis",
                line=line_opt,
                showscale=True,
            ),
        ),
        layout=layout,
    )

    return fig


group_names = {0: "No change", 1: "Addition", 2: "Deletion"}


def twod_distribution(x, groups=None):
    if groups is None:
        hist_data = [x]
        group_labels = ["Difference in Z"]  # name of the dataset
        fig = ff.create_distplot(hist_data, group_labels)
    else:
        groups_id = list(np.unique(groups))
        hist_data, group_labels = [], []
        for g in groups_id:
            hist_data.append(x[groups == g])
            group_labels.append(group_names[g])
        fig = ff.create_distplot(
            hist_data, group_labels, colors=["black", "blue", "red"]
        )
    return fig


def scatter3d(x, y, z, color):
    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=2,
            color=color,  # set color to an array/list of desired values
            colorscale="Rainbow",  # choose a colorscale
            opacity=0.8,
        ),
    )
    return scatter


def fig_3d(x, y, z, color):

    fig = go.Figure(data=[scatter3d(x, y, z, color)])

    return fig


def main():

    csv_file = sys.argv[1]

    # Read data from a csv
    table = pd.read_csv(csv_file)[["//X", "Y", "Z", "label_ch"]]
    table.columns = ["X", "Y", "Z", "label"]
    x, y, z = table.X.values, table.Y.values, table.Z.values
    fig = fig_3d(x, y, z, z)
    fig.show()
    fig = fig_3d(x, y, z, table.label.values)
    fig.show()


if __name__ == "__main__":
    main()
