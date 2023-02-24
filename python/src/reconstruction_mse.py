import sys
from utils_diff import load_data, predict_z
import pandas as pd
from opening_ply import ply_to_npy
import os
import numpy as np


def find_gt(dataname, path):
    if "one" in dataname:
        cl0 = "LyonSzero0.ply"
        cl1 = "LyonSzero1.ply"
    elif "two" in dataname:
        cl0 = "LyonStwo0.ply"
        cl1 = "LyonStwo1.ply"
    else:
        cl0 = "LyonSzero0.ply"
        cl1 = "LyonSzero0.ply"
    cl0 = os.path.join(path, cl0)
    cl1 = os.path.join(path, cl1)

    table0 = pd.DataFrame(ply_to_npy(cl0))
    table0.columns = ["X", "Y", "Z", "la"]

    table1 = pd.DataFrame(ply_to_npy(cl1))
    table1.columns = ["X", "Y", "Z", "la"]

    xmed = np.median(np.concatenate([table0[:, "X"], table1[:, "X"]], axis=0))
    ymed = np.median(np.concatenate([table0[:, "Y"], table1[:, "Y"]], axis=0))

    table0[:, "X"] = table0[:, "X"] - xmed
    table1[:, "X"] = table1[:, "X"] - xmed

    table0[:, "Y"] = table0[:, "Y"] - ymed
    table1[:, "Y"] = table1[:, "Y"] - ymed

    return table0, table1


(
    table0,
    table1,
    model0,
    model1,
    nv0,
    nv1,
    time0,
    time1,
    dataname,
    normalize,
    fs,
    method,
    lambda_t,
) = load_data(sys)

gt0, gt1 = find_gt(dataname, sys.argv[-2])

xy_onz0 = gt0[["X", "Y"]].values.astype("float32")
xy_onz1 = gt1[["X", "Y"]].values.astype("float32")
z0 = gt0[["Z"]].values.astype("float32")
z1 = gt1[["Z"]].values.astype("float32")

z0_pred = predict_z(
    model0,
    nv0,
    xy_onz0,
    normalize=normalize,
    time=time0,
)
mse0 = ((z0[:, 0] - z0_pred) ** 2).mean()

z1_pred = predict_z(
    model1,
    nv1,
    xy_onz1.copy(),
    normalize=normalize,
    time=time1,
)

mse1 = ((z1[:, 0] - z1_pred) ** 2).mean()

name_csv = f"{method}_{dataname}_reconstruction_results.csv"

scores = {
    "method": method,
    "normalize": normalize,
    "fs": fs,
    "MSE_PC0_GT": mse0,
    "MSE_PC1_GT": mse1,
}

pd.DataFrame(scores, index=[dataname]).to_csv(name_csv)
