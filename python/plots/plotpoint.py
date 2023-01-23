import sys
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

def scatter3d(x, y, z, color):
    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=color,                # set color to an array/list of desired values
            colorscale='Rainbow',   # choose a colorscale
            opacity=0.8
        )
    )
    return scatter

def fig_3d(x, y, z, color):


    fig = go.Figure(data=[scatter3d(x, y, z, color)])

    return fig

def main():

    csv_file = sys.argv[1]

    # Read data from a csv
    table = pd.read_csv(csv_file)[['//X', 'Y', 'Z', 'label_ch']]
    table.columns = ['X', 'Y', 'Z', 'label']
    x, y, z = table.X.values, table.Y.values, table.Z.values
    fig = fig_3d(x, y, z, z)
    fig.show()
    fig = fig_3d(x, y, z, table.label.values)
    fig.show()
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
