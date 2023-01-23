import numpy as np
import pandas as pd
import argparse
from plotly.subplots import make_subplots
from plotpoint import scatter3d

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
    return npz["indices"], npz["z1"], npz["z2"], npz["labels"]

def main():
    opt = parser_f()
    table1 = load_csv(opt.csv1)
    table2 = load_csv(opt.csv2)
    grid, z1, z2, label = load_npz(opt.npz)
    specs = [[{'is_3d':True} for _ in range(3)] for _ in range(3)]
    subplot_titles = ("Cloud1", "(hat) Cloud1", "", "Cloud2", "(hat) Cloud2", "(hat) (Cloud2-Cloud1)", "Cloud2 GT", "(hat) Cloud2 GT", "(hat) (Cloud2-Cloud1) GT")

    fig = make_subplots(rows=3, cols=3, specs=specs, subplot_titles=subplot_titles)

    diff = z2 - z1
    idx = np.where(~np.isnan(label))[0]
    fig.add_trace(scatter3d(table1.X, table1.Y, table1.Z, table1.Z),
                row=1, col=1)

    fig.add_trace(scatter3d(table2.X, table2.Y, table2.Z, table2.Z),
                row=2, col=1)

    fig.add_trace(scatter3d(table2.X, table2.Y, table2.Z, table2.label),
                row=3, col=1)
    
    fig.add_trace(scatter3d(grid[idx,0], grid[idx,1], z1[idx], z1[idx]),
                row=1, col=2)

    fig.add_trace(scatter3d(grid[idx,0], grid[idx,1], z2[idx], z2[idx]),
                row=2, col=2)

    fig.add_trace(scatter3d(grid[idx,0], grid[idx,1], z2[idx], label[idx]),
                row=3, col=2)

    fig.add_trace(scatter3d(grid[idx,0], grid[idx,1], diff[idx], diff[idx]),
                row=2, col=3)

    fig.add_trace(scatter3d(grid[idx,0], grid[idx,1], diff[idx], label[idx]),
                row=3, col=3)

    fig.show()



if __name__ == "__main__":
    main()
