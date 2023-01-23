import numpy as np
import pandas as pd
import argparse
from plotly.subplots import make_subplots
from plotpoint import scatter3d
import plotly.graph_objects as go

def parser_f():

    parser = argparse.ArgumentParser(
        description="Train supervised NN on cell",
    )
    parser.add_argument(
        "--npz",
        type=str,
    )
    parser.add_argument(
        "--csv1",
        type=str,
    )
    parser.add_argument(
        "--csv2",
        type=str,
    )
    args = parser.parse_args()
    return args

def load_csv(file):
    table = pd.read_csv(file)[['//X', 'Y', 'Z', 'label_ch']]
    table.columns = ['X', 'Y', 'Z', 'label']
    return table 

def load_npz(file):
    npz = np.load(file)
    return npz["indices"], npz["z1_ongrid"], npz["z2_ongrid"], npz["labels_ongrid"]



def contour(x, y, z, color):
    df = pd.DataFrame(np.vstack([x,y,z, color]).T)
    df.columns = ['x', 'y', 'z', 'color']
    df.x = df.x.astype(int)
    df.y = df.y.astype(int)
    sq_z = df.pivot_table(values='z', index='x', columns='y')
    sq_z = sq_z.interpolate(method="linear", limit_direction="backward", limit_area='inside')
    cont = go.Contour(
           z=sq_z,
            x=sq_z.index, # horizontal axis
            y=sq_z.columns, # vertical axis
           colorbar=dict(nticks=10, ticks='outside',
                         ticklen=5, tickwidth=1,
                         showticklabels=True,
                         tickangle=0, tickfont_size=12)
            )
    return cont


def treat_image(x, y, z, color):
    df = pd.DataFrame(np.vstack([x,y,z, color]).T)
    df.columns = ['x', 'y', 'z', 'color']
    df.x = df.x.astype(int)
    df.y = df.y.astype(int)
    sq_z = df.pivot_table(values='z', index='x', columns='y')
    sq_z = sq_z.interpolate(method="linear", limit_direction="backward", limit_area='inside')
    def func(x):
        try:
            return int(x)
        except:
            return x
    for el in sq_z.columns:
        sq_z[el] = sq_z[el].apply(func)
    sq_z = sq_z.fillna(0).astype(int)
    sq_z[sq_z == 2] = -1
    img = go.Heatmap(z=sq_z.values, x=sq_z.index, y=sq_z.columns)
    
    return img

def main():
    opt = parser_f()
    table1 = load_csv(opt.csv1)
    table2 = load_csv(opt.csv2)
    grid, z1, z2, label = load_npz(opt.npz)
    # specs = [[{'is_3d':True} for _ in range(3)] for _ in range(3)]
    subplot_titles = ("(hat) Cloud1", "(hat) Cloud2",  "Cloud2 GT", "(hat) (Cloud2-Cloud1)")

    fig = make_subplots(rows=1, cols=4, shared_xaxes=True, shared_yaxes=True, subplot_titles=subplot_titles)

    diff = z2 - z1
    idx = np.where(~np.isnan(label))[0]
    # fig.add_trace(contour(table1.X, table1.Y, table1.Z, table1.Z),
    #             row=1, col=1)

    # fig.add_trace(scatter3d(table2.X, table2.Y, table2.Z, table2.Z),
    #             row=2, col=1)

    # fig.add_trace(scatter3d(table2.X, table2.Y, table2.Z, table2.label),
    #             row=3, col=1)
    
    fig.add_trace(contour(grid[idx,0], grid[idx,1], z1[idx], z1[idx]),
                row=1, col=1)

    fig.add_trace(contour(grid[idx,0], grid[idx,1], z2[idx], z2[idx]),
                row=1, col=2)

    fig.add_trace(treat_image(grid[idx,0], grid[idx,1], label[idx], z2[idx] ),
                row=1, col=3)

    fig.add_trace(contour(grid[idx,0], grid[idx,1], diff[idx], diff[idx]),
                row=1, col=4)

    # fig.add_trace(contour(grid[idx,0], grid[idx,1], diff[idx], label[idx]),
    #             row=3, col=3)
    fig.update_xaxes(matches='x', showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(matches='y', showticklabels=False, showgrid=False, zeroline=False)
    fig.show()



if __name__ == "__main__":
    main()
