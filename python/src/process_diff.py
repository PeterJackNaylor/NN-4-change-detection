import sys
import numpy as np
from utils_diff import load_data, predict_z
from utils import compute_iou, compute_auc_mc, compute_mse, gmm_predict
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
label = "label" in table1.columns
if label:
    y_on1 = table1["label"].values

    # compute IoU
    iou_b, thresh_b, pred = compute_iou(diff_z_on1, y_on1)
    iou_gmm, thresh_gmm, pred_gmm = compute_iou(diff_z_on1, y_on1, use_gmm=True)
    auc_score = compute_auc_mc(diff_z_on1, y_on1)
    print("best threshold:", iou_b)
    print("threshold with gmm", iou_gmm)
    print("auc_scores:", auc_score)
else:
    pred_gmm, thresh_gmm = gmm_predict(diff_z_on1)

mse0 = compute_mse(z0_on0, table0[["Z"]].values[:, 0])
mse1 = compute_mse(z1_on1, table1[["Z"]].values[:, 0])

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
sub_z1_on1 = z1_on1[idx]
sub_z0_on1 = z0_on1[idx]
sub_pred_gmm = pred_gmm[idx]

if label:
    sub_y_on1 = y_on1[idx]
    sub_pred = pred[idx]

try:
    fig = twod_distribution(sub_diff_z_on1)
    name_png = f"{dataname}_diffZ1_distribution.png"
    fig.write_image(name_png)
except ValueError:
    pass

if label:
    try:
        fig = twod_distribution(sub_diff_z_on1, sub_y_on1)
        name_png = f"{dataname}_diffZ1_distribution_by_label.png"
        fig.write_image(name_png)
    except ValueError:
        pass


try:
    fig = twod_distribution(sub_diff_z_on1, sub_pred_gmm)
    name_png = f"{dataname}_diffZ1_distribution_by_gmmlabel.png"
    fig.write_image(name_png)
except ValueError:
    pass

try:
    name_png = f"{dataname}_diffZ1.png"
    fig = scatter2d(sub_X, sub_Y, sub_diff_z_on1)
    fig.write_image(name_png)
except ValueError:
    pass

if label:
    try:
        name_png = f"{dataname}_diffZ1_and_label.png"
        fig = scatter2d(sub_X, sub_Y, sub_diff_z_on1, sub_y_on1)
        fig.write_image(name_png)
    except ValueError:
        pass

    try:
        name_png = f"{dataname}_diffZ1_and_prediction.png"
        fig = scatter2d(sub_X, sub_Y, sub_diff_z_on1, sub_pred)
        fig.write_image(name_png)
    except ValueError:
        pass

try:
    name_png = f"{dataname}_diffZ1_and_gmmprediction.png"
    fig = scatter2d(sub_X, sub_Y, sub_diff_z_on1, sub_pred_gmm)
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
    fig = scatter2d(sub_X, sub_Y, sub_Z, sub_pred_gmm)
    name_png = f"{dataname}_Z1_and_gmmprediction.png"
    fig.write_image(name_png)
except ValueError:
    pass

if label:
    try:
        fig = scatter2d(sub_X, sub_Y, sub_Z, sub_y_on1)
        name_png = f"{dataname}_Z1_and_label.png"
        fig.write_image(name_png)
    except ValueError:
        pass

    try:
        fig = scatter2d(sub_X, sub_Y, sub_Z, sub_pred)
        name_png = f"{dataname}_Z1_and_prediction.png"
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
    fig = scatter2d(sub_X, sub_Y, sub_z1_on1, sub_pred_gmm)
    name_png = f"{dataname}_predictionZ1_and_gmmprediction.png"
    fig.write_image(name_png)
except ValueError:
    pass

if label:
    try:
        fig = scatter2d(sub_X, sub_Y, sub_z1_on1, sub_y_on1)
        name_png = f"{dataname}_predictionZ1_and_label.png"
        fig.write_image(name_png)
    except ValueError:
        pass

    try:
        fig = scatter2d(sub_X, sub_Y, sub_z1_on1, sub_pred)
        name_png = f"{dataname}_predictionZ1_and_prediction.png"
        fig.write_image(name_png)
    except ValueError:
        pass


try:
    fig = scatter2d(sub_X, sub_Y, sub_z0_on1)
    name_png = f"{dataname}_predictionZ0.png"
    fig.write_image(name_png)
except ValueError:
    pass

try:
    fig = scatter2d(sub_X, sub_Y, sub_z0_on1, sub_pred_gmm)
    name_png = f"{dataname}_predictionZ0_and_gmmprediction.png"
    fig.write_image(name_png)
except ValueError:
    pass

if label:
    try:
        fig = scatter2d(sub_X, sub_Y, sub_z0_on1, sub_y_on1)
        name_png = f"{dataname}_predictionZ0_and_label.png"
        fig.write_image(name_png)
    except ValueError:
        pass

    try:
        fig = scatter2d(sub_X, sub_Y, sub_z0_on1, sub_pred)
        name_png = f"{dataname}_predictionZ0_and_prediction.png"
        fig.write_image(name_png)
    except ValueError:
        pass


name_csv = f"{method}_{dataname}_results.csv"

scores = {
    "method": method,
    "normalize": normalize,
    "fs": fs,
    "threshold_gmm_low": thresh_gmm[0],
    "threshold_gmm_high": thresh_gmm[1],
    "len(threshold_gmm)": len(thresh_gmm),
    "MSE_PC0": mse0,
    "MSE_PC1": mse1,
}
if label:
    scores["AUC_nochange"] = auc_score["No change"]
    scores["AUC_addition"] = auc_score["Addition"]
    scores["AUC_deletion"] = auc_score["Deletion"]
    scores["IoU_gmm"] = iou_gmm
    scores["IoU_best"] = iou_b
    scores["threshold_best_low"] = thresh_b[0]
    scores["threshold_best_high"] = thresh_b[1]
pd.DataFrame(scores, index=[dataname]).to_csv(name_csv)
