import plotly.graph_objects as go
import sys
import pandas as pd
import numpy as np

def plot_surface(xmin, xmax, ymin, ymax, z):

    x, y = np.linspace(xmin, xmax, (xmax-xmin+1)/2), np.linspace(ymin, ymax, (ymax-ymin+1)/2)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[xmin, xmax],),
            yaxis = dict(nticks=4, range=[ymin,ymax],),
            zaxis = dict(nticks=4, range=[z.min(),z.max()+10],),),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10)
    )
    fig.show()

def plot_tri_grid(x, y, z):

    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
    return fig


def main():

    csv_file = sys.argv[1]

    # Read data from a csv
    table = pd.read_csv(csv_file)[['//X', 'Y', 'Z']]
    table.columns = ['X', 'Y', 'Z']
    x, y, z = table.X.values, table.Y.values,table.Z.values
    fig = plot_tri_grid(x, y, z)
    fig.show()


if __name__ == "__main__":
    main()
