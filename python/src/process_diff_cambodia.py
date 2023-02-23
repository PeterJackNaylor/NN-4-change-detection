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

mse0 = compute_mse(z0_on0, table0[["Z"]].values[:, 0])
mse1 = compute_mse(z1_on1, table1[["Z"]].values[:, 0])

print("MSE PC0:", mse0)
print("MSE PC1:", mse1)

size0 = table0.X.shape[0]
size1 = table1.X.shape[0]

factor = 1

idx = np.arange(size1)
np.random.shuffle(idx)
idx = idx[: size1 // factor]

sub_X = table1.X.values[idx]
sub_Y = table1.Y.values[idx]
sub_Z = table1.Z.values[idx]
sub_diff_z_on1 = diff_z_on1[idx]
sub_z1_on1 = z1_on1[idx]
sub_z0_on1 = z0_on1[idx]

try:
    fig = twod_distribution(sub_diff_z_on1)
    name_png = f"{dataname}_diffZ1_distribution.png"
    fig.write_image(name_png)
except ValueError and np.linalg.LinAlgError:
    pass


try:
    name_png = f"{dataname}_diffZ1.png"
    fig = scatter2d(sub_X, sub_Y, sub_diff_z_on1)
    fig.write_image(name_png)
except ValueError:
    pass

try:
    idx0 = np.arange(size0)
    np.random.shuffle(idx0)
    idx0 = idx0[: size0 // factor]

    sub_X0 = table0.X.values[idx0]
    sub_Y0 = table0.Y.values[idx0]
    sub_Z0 = table0.Z.values[idx0]
    fig = scatter2d(sub_X0, sub_Y0, sub_Z0)
    name_png = f"{dataname}_Z0.png"
    fig.write_image(name_png)
except ValueError:
    pass


try:
    fig = scatter2d(sub_X, sub_Y, sub_Z)
    name_png = f"{dataname}_Z1.png"
    fig.write_image(name_png)
except ValueError:
    pass

try:
    fig = scatter2d(sub_X, sub_Y, sub_z1_on1)
    name_png = f"{dataname}_predictionZ1.png"
    fig.write_image(name_png)
except ValueError:
    pass

try:
    fig = scatter2d(sub_X, sub_Y, sub_z0_on1)
    name_png = f"{dataname}_predictionZ0.png"
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
