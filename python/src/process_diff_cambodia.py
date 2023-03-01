import sys
import numpy as np
from utils_diff import load_data, predict_z
from utils import compute_mse
from plotpoint import scatter2d, twod_distribution
import pandas as pd


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

xy_onz0 = table0[["X", "Y"]].values.astype("float32")
xy_onz1 = table1[["X", "Y"]].values.astype("float32")

z0_on0 = predict_z(
    model0,
    nv0,
    xy_onz0,
    normalize=normalize,
    time=time0,
)
z0_on1 = predict_z(
    model0,
    nv0,
    xy_onz1.copy(),
    normalize=normalize,
    time=time0,
)
z1_on1 = predict_z(
    model1,
    nv1,
    xy_onz1.copy(),
    normalize=normalize,
    time=time1,
)

diff_z_on1 = z1_on1 - z0_on1
diff_z_on1 = np.nan_to_num(diff_z_on1)

print("Filtering positive change")
diff_z_on1[diff_z_on1 > 0] = 0
M = diff_z_on1.min()

bins = [
    0,
    0.1 * M,
    0.2 * M,
    0.3 * M,
    0.4 * M,
    0.5 * M,
    0.6 * M,
    0.7 * M,
    0.8 * M,
    0.9 * M,
    M,
]
names = list(range(len(bins) - 1))
names.reverse()
bins.reverse()
table1["Cat_diff"] = pd.cut(diff_z_on1, bins, labels=names)
table1[["X", "Y", "Z", "Cat_diff"]].to_csv(f"{dataname}_xyz_{method}_change.csv")


mse0 = compute_mse(z0_on0, table0[["Z"]].values[:, 0])
mse1 = compute_mse(z1_on1, table1[["Z"]].values[:, 0])

print("MSE PC0:", mse0)
print("MSE PC1:", mse1)

size0 = table0.X.shape[0]
size1 = table1.X.shape[0]


idx = np.arange(size1)
xmax, xmin = table1.X.values.max(), table1.X.values.min()
ymax, ymin = table1.Y.values.max(), table1.Y.values.min()
xmid = (xmin + xmax) / 2
ymid = (ymin + ymax) / 2
table1 = table1.reset_index(drop=True)
idx_bl = idx[(table1.X < xmid) & (table1.Y < ymid)]
idx_br = idx[(table1.X >= xmid) & (table1.Y < ymid)]
idx_tl = idx[(table1.X < xmid) & (table1.Y >= ymid)]
idx_tr = idx[(table1.X >= xmid) & (table1.Y >= ymid)]

idx0 = np.arange(size0)
table0 = table0.reset_index(drop=True)
idx0_bl = idx0[(table0.X < xmid) & (table0.Y < ymid)]
idx0_br = idx0[(table0.X >= xmid) & (table0.Y < ymid)]
idx0_tl = idx0[(table0.X < xmid) & (table0.Y >= ymid)]
idx0_tr = idx0[(table0.X >= xmid) & (table0.Y >= ymid)]


index_names = {
    "top_left": idx_tl,
    "top_right": idx_tr,
    "bottom_left": idx_bl,
    "bottom_right": idx_br,
}

index0_names = {
    "top_left": idx0_tl,
    "top_right": idx0_tr,
    "bottom_left": idx0_bl,
    "bottom_right": idx0_br,
}

size = 2

for key, idx in index_names.items():
    sub_X = table1.X.values[idx]
    sub_Y = table1.Y.values[idx]
    sub_Z = table1.Z.values[idx]
    sub_diff_z_on1 = diff_z_on1[idx]
    sub_z1_on1 = z1_on1[idx]
    sub_z0_on1 = z0_on1[idx]

    try:
        fig = twod_distribution(sub_diff_z_on1)
        name_png = f"{key}_{dataname}_diffZ1_distribution.png"
        fig.write_image(name_png)
    except ValueError and np.linalg.LinAlgError:
        pass

    try:
        name_png = f"{key}_{dataname}_diffZ1.png"
        fig = scatter2d(sub_X, sub_Y, sub_diff_z_on1, size=size)
        fig.write_image(name_png)
    except ValueError:
        pass

    try:
        idx0_pos = index0_names[key]
        sub_X0 = table0.X.values[idx0_pos]
        sub_Y0 = table0.Y.values[idx0_pos]
        sub_Z0 = table0.Z.values[idx0_pos]
        fig = scatter2d(sub_X0, sub_Y0, sub_Z0, size=size)
        name_png = f"{key}_{dataname}_Z0.png"
        fig.write_image(name_png)
    except ValueError:
        pass

    try:
        fig = scatter2d(sub_X, sub_Y, sub_Z, size=size)
        name_png = f"{key}_{dataname}_Z1.png"
        fig.write_image(name_png)
    except ValueError:
        pass

    try:
        fig = scatter2d(sub_X, sub_Y, sub_z1_on1, size=size)
        name_png = f"{key}_{dataname}_predictionZ1.png"
        fig.write_image(name_png)
    except ValueError:
        pass

    try:
        fig = scatter2d(sub_X, sub_Y, sub_z0_on1, size=size)
        name_png = f"{key}_{dataname}_predictionZ0.png"
        fig.write_image(name_png)
    except ValueError:
        pass


name_csv = f"{method}_{dataname}_results.csv"

scores = {
    "method": method,
    "normalize": normalize,
    "fs": fs,
    "MSE_PC0": mse0,
    "MSE_PC1": mse1,
}

pd.DataFrame(scores, index=[dataname]).to_csv(name_csv)
