
import numpy as np
from glob import glob
import sys
from utils import compute_iou
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm

def open_npz_compute(f, OT=False):
    file = np.load(f)
    gt = file["labels_on1"]
    th = file["thresh_bin"]
    if OT:
        z = file["changes"]
        # opened_z = file["opened_changes"]
    else:
        z = file["z1_on1"] - file["z0_on1"]
        # opened_z = z.copy()
    pred = np.zeros_like(z).astype(int)
    pred[z > th] = 1
    pred[z < -th] = 1
    iou = file["IoU_bin"]
    iou_mc = file["IoU_mc"]
    return z, gt, pred, iou, iou_mc#, opened_z

files = glob("*.npz")

dataname = sys.argv[1]
method = sys.argv[2]
is_OT = "OT" == method[:2]

diff_z, prediction, gt = [], [], []
diff_z_opened = []
iou_chunks, iou_mc_chunks, chunk_id, size = [], [], [], []
max_changes, min_changes, labels = [], [], []
acc, pearson = [], []

print("Reading files")
for f in tqdm(files):
    # z_f, gt_f, yhat, iou, iou_mc, z_f_opened = open_npz_compute(f, is_OT)
    z_f, gt_f, yhat, iou, iou_mc = open_npz_compute(f, is_OT)
    diff_z.append(z_f)
    prediction.append(yhat)
    gt.append(gt_f)
    iou_chunks.append(iou)
    iou_mc_chunks.append(iou_mc)
    chunk_id.append(f.split("-")[0])
    max_changes.append(z_f.max())
    min_changes.append(z_f.min())
    size.append(z_f.shape[0])
    labels.append((gt_f != 0).sum())
    acc.append(accuracy_score(gt_f, yhat))
    pearson.append(pearsonr(gt_f, yhat)[0])
    # diff_z_opened.append(z_f_opened)


tmp_table = pd.DataFrame(
    {"chunk_id": chunk_id,
    "iou": iou_chunks,
    "iou_mc": iou_mc_chunks,
    "max_changes": max_changes,
    "min_changes": min_changes,
    "nlabels": labels,
    "size": size,
    "acc": acc,
    "pearson": pearson
    }
)

tmp_table.to_csv("{}_{}_chunkinfo.csv".format(dataname, method), index=False)


diff_z = np.concatenate(diff_z, axis=0)
prediction = np.concatenate(prediction, axis=0)
# diff_z_opened = np.concatenate(diff_z_opened, axis=0)
gt = np.concatenate(gt, axis=0)



table = pd.DataFrame()
print("Computing iou")
iou_bin, _, _, iou_mc, _, _ = compute_iou(diff_z, gt)
iou_bin_fix_th, _, _, iou_mc_fix_th, _, _ = compute_iou(diff_z, gt, threshold=5)

# iou_opened, _, _, iou_mc_opened, _, _ = compute_iou(diff_z_opened, gt)
# iou_opened_fix_th, _, _, iou_mc_opened_fix_th, _, _ = compute_iou(diff_z_opened, gt, threshold=5)

table.loc[dataname, "iou_mc"] = iou_mc
table.loc[dataname, "iou"] = iou_bin
table.loc[dataname, "iou_mc_threshold"] = iou_mc_fix_th
table.loc[dataname, "iou_threshold"] = iou_bin_fix_th

# table.loc[dataname, "iou_mc_opened_threshold"] = iou_mc_opened_fix_th
# table.loc[dataname, "iou_opened_threshold"] = iou_opened_fix_th
# table.loc[dataname, "iou_mc_opened"] = iou_mc_opened
# table.loc[dataname, "iou_opened"] = iou_opened
table.loc[dataname, "method"] = method
table.to_csv("{}_{}.csv".format(dataname, method))
