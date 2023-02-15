import sys
import numpy as np
from utils_diff import load_csv_weight_npz, predict_z
from utils import compute_iou, compute_auc_mc, compute_mse
from plotpoint import scatter2d, twod_distribution
from parser import read_yaml
import pandas as pd

yaml_file = read_yaml(sys.argv[-1])
normalize = yaml_file.norm
method = sys.argv[1]

if "double" in method:
    weight0 = sys.argv[2]
    weight1 = sys.argv[3]

    csv0 = sys.argv[4]
    csv1 = sys.argv[5]

    npz0 = sys.argv[6]
    npz1 = sys.argv[7]

    dataname = weight0.split("__")[0][:-1]
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

    table0, model0, nv0 = load_csv_weight_npz(
        csvfile0, None, w0_file, npz_0, name0, time
    )
    table1, model1, nv1 = load_csv_weight_npz(
        csvfile1, None, w1_file, npz_1, name1, time
    )
    four_opt = name0.split("FOUR=")[1].split("__")[0].split("_")[0]
    four_opt = four_opt == "--fourier"
else:

    weight = sys.argv[2]

    csvfile0 = sys.argv[3]
    csvfile1 = sys.argv[4]

    npz = sys.argv[5]
    time = 1

    dataname = weight.split("__")[0]
    name = weight.split(".p")[0]

    table, model, nv = load_csv_weight_npz(
        csvfile0,
        csvfile1,
        weight,
        npz,
        name,
        time,
    )
    table0 = table[table["T"] == 0]
    table1 = table[table["T"] == 1]
    model0, model1 = model, model
    nv0, nv1 = nv, nv
    time0 = 0.0
    time1 = 1.0
    four_opt = name.split("FOUR=")[1].split("__")[0].split("_")[0]
    four_opt = four_opt == "--fourier"


z0_n = table0.shape[0]
z1_n = table1.shape[0]

labels_1_n = (table1["label"].astype(int).values == 1).sum()
labels_2_n = (table1["label"].astype(int).values == 2).sum()

# grid_indices = define_grid(table0, table1, step=2)
# xy_grid = grid_indices.copy()  # .astype("float32")
xy_onz0 = table0[["X", "Y"]].values  # .astype("float32")
xy_onz1 = table1[["X", "Y"]].values  # .astype("float32")

z0_on0 = predict_z(model0, nv0, xy_onz0, normalize=normalize, time=time0)
z0_on1 = predict_z(model0, nv0, xy_onz1, normalize=normalize, time=time0)
z1_on1 = predict_z(model1, nv1, xy_onz1, normalize=normalize, time=time1)

diff_z_on1 = z1_on1 - z0_on1

# z0_ongrid = predict_z(model0, nv0, xy_grid, normalize=normalize, time=time0)
# z1_ongrid = predict_z(model1, nv1, xy_grid, normalize=normalize, time=time1)


# diff_z = z1_ongrid - z0_ongrid
y_on1 = table1["label"].values
# compute IoU

iou_b, thresh_b, pred = compute_iou(diff_z_on1, y_on1)
iou_gmm, thresh_gmm, pred_gmm = compute_iou(diff_z_on1, y_on1, use_gmm=True)

auc_score = compute_auc_mc(diff_z_on1, y_on1)

mse0 = compute_mse(z0_on0, table0[["Z"]].values[:, 0])
mse1 = compute_mse(z1_on1, table1[["Z"]].values[:, 0])

print("best threshold:", iou_b)
print("threshold with gmm", iou_gmm)
print("auc_scores:", auc_score)
print("MSE PC0:", mse0)
print("MSE PC1:", mse1)


size1 = table1.X.shape[0]

if size1 > 1e6:
    factor = 100
elif size1 > 5e5:
    factor = 20
elif size1 > 1e5:
    factor = 5
elif size1 > 5e4:
    factor = 2
elif size1 > 1e4:
    factor = 1
else:
    factor = 1

idx = np.arange(size1)
np.random.shuffle(idx)
idx = idx[: size1 // factor]

sub_X = table1.X.values[idx]
sub_Y = table1.Y.values[idx]
sub_Z = table1.Z.values[idx]
sub_diff_z_on1 = diff_z_on1[idx]
sub_y_on1 = y_on1[idx]
sub_z1_on1 = z1_on1[idx]
sub_z0_on1 = z0_on1[idx]
sub_pred = pred[idx]
sub_pred_gmm = pred_gmm[idx]

fig = twod_distribution(sub_diff_z_on1)
name_png = f"{dataname}_diffZ1_distribution.png"
fig.write_image(name_png)

fig = twod_distribution(sub_diff_z_on1, sub_y_on1)
name_png = f"{dataname}_diffZ1_distribution_by_label.png"
fig.write_image(name_png)

fig = twod_distribution(sub_diff_z_on1, sub_pred_gmm)
name_png = f"{dataname}_diffZ1_distribution_by_gmmlabel.png"
fig.write_image(name_png)

name_png = f"{dataname}_diffZ1.png"
fig = scatter2d(sub_X, sub_Y, sub_diff_z_on1)
fig.write_image(name_png)

name_png = f"{dataname}_diffZ1_and_label.png"
fig = scatter2d(sub_X, sub_Y, sub_diff_z_on1, sub_y_on1)
fig.write_image(name_png)

name_png = f"{dataname}_diffZ1_and_prediction.png"
fig = scatter2d(sub_X, sub_Y, sub_diff_z_on1, sub_pred)
fig.write_image(name_png)
name_png = f"{dataname}_diffZ1_and_gmmprediction.png"
fig = scatter2d(sub_X, sub_Y, sub_diff_z_on1, sub_pred_gmm)
fig.write_image(name_png)


fig = scatter2d(sub_X, sub_Y, sub_Z)
name_png = f"{dataname}_Z1.png"
fig.write_image(name_png)

fig = scatter2d(sub_X, sub_Y, sub_Z, sub_pred)
name_png = f"{dataname}_Z1_and_prediction.png"
fig.write_image(name_png)
fig = scatter2d(sub_X, sub_Y, sub_Z, sub_pred_gmm)
name_png = f"{dataname}_Z1_and_gmmprediction.png"
fig.write_image(name_png)

fig = scatter2d(sub_X, sub_Y, sub_Z, sub_y_on1)
name_png = f"{dataname}_Z1_and_label.png"
fig.write_image(name_png)

fig = scatter2d(sub_X, sub_Y, sub_z1_on1)
name_png = f"{dataname}_predictionZ1.png"
fig.write_image(name_png)

fig = scatter2d(sub_X, sub_Y, sub_z1_on1, sub_pred)
name_png = f"{dataname}_predictionZ1_and_prediction.png"
fig.write_image(name_png)
fig = scatter2d(sub_X, sub_Y, sub_z1_on1, sub_pred_gmm)
name_png = f"{dataname}_predictionZ1_and_gmmprediction.png"
fig.write_image(name_png)

fig = scatter2d(sub_X, sub_Y, sub_z1_on1, sub_y_on1)
name_png = f"{dataname}_predictionZ1_and_label.png"
fig.write_image(name_png)


fig = scatter2d(sub_X, sub_Y, sub_z0_on1)
name_png = f"{dataname}_predictionZ0.png"
fig.write_image(name_png)

fig = scatter2d(sub_X, sub_Y, sub_z0_on1, sub_pred)
name_png = f"{dataname}_predictionZ0_and_prediction.png"
fig.write_image(name_png)
fig = scatter2d(sub_X, sub_Y, sub_z0_on1, sub_pred_gmm)
name_png = f"{dataname}_predictionZ0_and_gmmprediction.png"
fig.write_image(name_png)

fig = scatter2d(sub_X, sub_Y, sub_z0_on1, sub_y_on1)
name_png = f"{dataname}_predictionZ0_and_label.png"
fig.write_image(name_png)


name_csv = f"{method}_{dataname}_results.csv"

scores = {
    "method": method,
    "normalize": normalize,
    "fourrier": four_opt,
    "IoU_gmm": iou_gmm,
    "threshold_gmm_low": thresh_gmm[0],
    "threshold_gmm_high": thresh_gmm[1],
    "len(threshold_gmm)": len(thresh_gmm),
    "IoU_best": iou_b,
    "threshold_best_low": thresh_b[0],
    "threshold_best_high": thresh_b[1],
    "AUC_nochange": auc_score["No change"],
    "AUC_addition": auc_score["Addition"],
    "AUC_deletion": auc_score["Deletion"],
    "MSE_PC0": mse0,
    "MSE_PC1": mse1,
}

pd.DataFrame(scores, index=[dataname]).to_csv(name_csv)
