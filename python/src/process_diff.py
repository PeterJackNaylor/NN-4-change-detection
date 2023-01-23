import sys
import numpy as np
from utils_diff import load_csv_weight_npz, define_grid, predict_z
from utils import compute_iou
from plotpoint import fig_3d
import pandas as pd

double = sys.argv[1]
if double == "double":
    weight0 = sys.argv[2]
    weight1 = sys.argv[3]

    csv0 = sys.argv[4]
    csv1 = sys.argv[5]

    npz0 = sys.argv[6]
    npz1 = sys.argv[7]

    dataname = weight0.split("__")[0]
    tag = int(weight0.split("__")[0][-1])

    n0 = weight0.split(".p")[0]
    n1 = weight1.split(".p")[0]

    if tag != 1:
        w0_file = weight0
        w1_file = weight1
        csvfile0 = csv0
        csvfile1 = csv1
        npz_0 = npz0
        npz_1 = npz1
        name0 = n0
        name1 = n1
    else:
        w0_file = weight1
        w1_file = weight0
        csvfile0 = csv1
        csvfile1 = csv0
        npz_0 = npz1
        npz_1 = npz0
        name0 = n1
        name1 = n0
    time = time0 = time1 = -1


    table0, model0, B0, nv0 = load_csv_weight_npz(csvfile0, None, w0_file, npz_0, name0, time)
    table1, model1, B1, nv1 = load_csv_weight_npz(csvfile1, None, w1_file, npz_1, name1, time)
else:

    weight = sys.argv[2]

    csvfile0 = sys.argv[3]
    csvfile1 = sys.argv[4]

    npz = sys.argv[5]
    time = 1

    dataname = weight.split("__")[0]
    name = weight.split(".p")[0]

    table, model, B, nv = load_csv_weight_npz(csvfile0, csvfile1, weight, npz, name, time)
    table0 = table[table["T"] == 0]
    table1 = table[table["T"] == 1]
    model0, model1 = model, model
    B0, B1 = B, B
    nv0, nv1 = nv, nv
    time0 = 0.
    time1 = 1.

z0_n = table0.shape[0]
z1_n = table1.shape[0]

labels_1_n = (table1["label"].astype(int).values == 1).sum()
labels_2_n = (table1["label"].astype(int).values == 2).sum()

grid_indices = define_grid(table0, table1, step=2)
xy_grid = grid_indices.copy()#.astype("float32")
xy_onz1 = table1[['X', 'Y']].values#.astype("float32")

z0_on1 = predict_z(model0, B0, nv0, xy_onz1, time=time0)
z1_on1 = predict_z(model1, B1, nv1, xy_onz1, time=time1)

diff_z_on1 = z1_on1 - z0_on1

z0_ongrid = predict_z(model0, B0, nv0, xy_grid, time=time0)
z1_ongrid = predict_z(model1, B1, nv1, xy_grid, time=time1)



diff_z = z1_ongrid - z0_ongrid
y_on1 = table1["label"].values
# compute IoU
bin_score, mc_score = compute_iou(diff_z_on1, y_on1)
iou_bin, thresh_bin, pred_bin = bin_score
iou_mc, thresh_mc, pred_mc = mc_score



table1_copy = table1.copy()
table1_copy.X = table1_copy.X.round().astype("int")
table1_copy.Y = table1_copy.Y.round().astype("int")
idx = table1_copy[["X", "Y"]].drop_duplicates().index
table1_copy = table1_copy.loc[idx]

grid_pd = pd.DataFrame(grid_indices).astype(int)
grid_pd.columns = ["X", "Y"]
result = pd.merge(grid_pd, table1_copy, on=["X", "Y"], how="left")
# result.label = result.label.fill_na(0)


size1 = table1.X.shape[0]

if size1 > 1e6:
    factor = 10
elif size1 > 5e5:
    factor = 5
elif size1 > 1e5:
    factor = 1
elif size1 > 5e4:
    factor = 1
elif size1 > 1e4:
    factor = 1
else: 
    factor = 1

idx = np.arange(size1)
np.random.shuffle(idx)
idx = idx[:size1 // factor]

sub_X = table1.X.values[idx]
sub_Y = table1.X.values[idx]
sub_Z = table1.Z.values[idx]
sub_diff_z_on1 = diff_z_on1[idx]
sub_y_on1 = y_on1[idx]
sub_z1_on1 = z1_on1[idx]
sub_z0_on1 = z0_on1[idx]
sub_pred_mc = pred_mc[idx]

fig = fig_3d(sub_X, sub_Y, sub_diff_z_on1, sub_y_on1)
name_png = f"diff_{dataname}_labels_on_z1.png"
fig.write_image(name_png)

fig = fig_3d(sub_X, sub_Y, sub_diff_z_on1, sub_pred_mc)
name_png = f"diff_{dataname}_pred_on_z1.png"
fig.write_image(name_png)

fig = fig_3d(sub_X, sub_Y, sub_Z, sub_pred_mc)
name_png = f"{dataname}_z1_on_z1.png"
fig.write_image(name_png)

fig = fig_3d(table1.X, table1.Y, sub_z1_on1, sub_pred_mc)
name_png = f"{dataname}_predz1_on_z1.png"
fig.write_image(name_png)

fig = fig_3d(table1.X, table1.Y, sub_z0_on1, sub_pred_mc)
name_png = f"{dataname}_predz0_on_z1.png"
fig.write_image(name_png)


fig = fig_3d(grid_indices[:,0], grid_indices[:,1], diff_z, result.label)
name_png = f"diff_{dataname}_labels.png"
fig.write_image(name_png)

fig = fig_3d(grid_indices[:,0], grid_indices[:,1], diff_z, diff_z)
name_png = f"diff_{dataname}.png"
fig.write_image(name_png)

# compute IoU

name_npz = f"{double}_{dataname}_results.npz"

np.savez(name_npz, indices=grid_indices, 
    x1=table1.X.values, y1=table1.Y.values,
    z0_on1=z0_on1,  z1_on1=z1_on1, 
    z0_ongrid=z0_ongrid, z1_ongrid=z1_ongrid, 
    labels_on1=y_on1, labels_ongrid=result.label,
    IoU_mc=iou_mc, thresh_mc=thresh_mc,
    IoU_bin=iou_bin, thresh_bin=thresh_bin,
    z0_n=z0_n, z1_n=z1_n, labels_1_n=labels_1_n, labels_2_n=labels_2_n)

