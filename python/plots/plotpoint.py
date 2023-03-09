import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np


def scatter2d(x, y, z, color=None, size=8, color_range=[], ignore_middle=False):
    xrange = [x.min(), x.max()]
    yrange = [y.min(), y.max()]
    if ignore_middle:
        idx = np.abs(z) > 2
        z = z.copy()[idx]
        x = x.copy()[idx]
        y = y.copy()[idx]

    if color is not None:
        if ignore_middle:
            color = color.copy()[idx]
        size += 2
        line_opt = {
            "color": color,
            "width": 2,
            "colorscale": ["rgba(0,0,0,0)", "blue", "red"],
        }
    else:
        line_opt = {"width": 0}
    # fig = go.Figure(layout=layout)
    layout = go.Layout(
        autosize=True,
        xaxis={"title": "x-axis", "visible": False, "showticklabels": False},
        yaxis={"title": "y-axis", "visible": False, "showticklabels": False},
        width=1000,
        height=1000,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
    )
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
                showscale=False,
                cmin=color_range[0],
                cmax=color_range[1],
            ),
        ),
        layout=layout,
    )
    fig.update_layout(xaxis_range=xrange)
    fig.update_layout(yaxis_range=yrange)
    # fig.update_traces(showscale=False)
    # fig.update_coloraxes(showscale=False)
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
