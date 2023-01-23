
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def compute_jaccard(ytrue, yhat):
    cm = confusion_matrix(ytrue, yhat, labels=[0,1,2])
    tp = cm[1:,1:].sum()
    jaccard_bin = tp / (cm[0,1:].sum() + cm[1:,0].sum() + tp)
    jaccard_mc_1 = cm[1,1] / (cm[1,:].sum() + cm[:,1].sum() - cm[1,1])
    jaccard_mc_2 = cm[2,2] / (cm[2,:].sum() + cm[:,2].sum() - cm[2,2])
    return jaccard_bin, np.mean([jaccard_mc_1, jaccard_mc_2])


def compute_iou(diffz, y, threshold=None):
    best_iou, best_miou = 0, 0
    best_t, best_mt = 0, 0
    fpred, fmpred = np.zeros_like(y), np.zeros_like(y)

    if threshold:
        t_range = [threshold]
    else:
        std = np.std(diffz)
        t_range = (np.arange(0, diffz.max(), step=std / 10))
    for thresh in tqdm(t_range):
        y_pred = np.zeros_like(y)
        y_pred[diffz > thresh] = 1
        y_pred[diffz < -thresh] = 2
        iou, miou = compute_jaccard(y, y_pred)
        if iou > best_iou:
            best_iou = iou
            best_t = thresh
            fpred = y_pred > 0
        if miou > best_miou:
            best_miou = miou
            best_mt = thresh
            fmpred = y_pred

    return best_iou, best_t, fpred, best_miou, best_mt, fmpred
