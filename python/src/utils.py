import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture as gmm
from sklearn.metrics import roc_auc_score


def compute_jaccard(ytrue, yhat):
    cm = confusion_matrix(ytrue, yhat, labels=[0, 1, 2])
    tp = cm[1:, 1:].sum()
    jaccard_bin = tp / (cm[0, 1:].sum() + cm[1:, 0].sum() + tp)
    jaccard_mc_1 = cm[1, 1] / (cm[1, :].sum() + cm[:, 1].sum() - cm[1, 1])
    jaccard_mc_2 = cm[2, 2] / (cm[2, :].sum() + cm[:, 2].sum() - cm[2, 2])
    return jaccard_bin, np.mean([jaccard_mc_1, jaccard_mc_2])


def determine_mapping_and_map(Z, gm):
    mapper_v = np.array([0, 1, 2])
    for i in range(gm.means_.shape[0]):
        if max(gm.means_) == gm.means_[i]:
            mapper_v[i] = 1
        elif min(gm.means_) == gm.means_[i]:
            mapper_v[i] = 2
        else:
            mapper_v[i] = 0
    return mapper_v[Z]


def gmm_predict(Z):

    mean_init = np.array([0, -15, 15]).reshape(-1, 1)
    weights_init = np.array([0.5, 0.25, 0.25])
    precisions_init = np.array([0.1, 0.01, 0.01]).reshape(-1, 1, 1)
    gm = gmm(
        n_components=3,
        means_init=mean_init,
        precisions_init=precisions_init,
        weights_init=weights_init,
    ).fit(
        Z.reshape(-1, 1)
    )  #
    pred_mc_gmm = gm.predict(Z.reshape(-1, 1))
    pred_mc_gmm = determine_mapping_and_map(pred_mc_gmm, gm)

    threshold = det_threshold(Z, pred_mc_gmm)
    print(gm.means_)
    print(threshold)
    return pred_mc_gmm, threshold


def det_threshold(Z, pred_mc):
    idx = Z.argsort()
    Zt = Z[idx]
    pt = pred_mc[idx]
    old_pred = pt[0]
    t = []
    for i in range(Zt.shape[0]):
        if old_pred != pt[i]:
            t.append(Zt[i])
            old_pred = pt[i]
    return t


def iou(y, diffz, thresh):
    y_pred = np.zeros_like(y)
    y_pred[diffz > thresh[1]] = 1
    y_pred[diffz < thresh[0]] = 2
    _, miou = compute_jaccard(y, y_pred)
    return y_pred, miou


def compute_iou(diffz, y, use_gmm=False, threshold=None):
    best_miou = 0
    best_mt = 0
    fmpred = np.zeros_like(y)

    if use_gmm:
        y_pred, threshold = gmm_predict(diffz)
        threshold[0] -= 1
        threshold[1] += 1
        y_pred, miou = iou(y, diffz, threshold)
        return miou, threshold, y_pred

    if threshold:
        y_pred, miou = iou(y, diffz, threshold)
        return miou, threshold, y_pred
    else:
        std = np.std(diffz)
        maxdiff = diffz.max()
        if maxdiff:
            t_range = np.arange(0, diffz.max(), step=std / 10)
        else:
            t_range = [0]
    for thresh in tqdm(t_range):
        y_pred, miou = iou(y, diffz, [-thresh, thresh])
        if miou > best_miou:
            best_miou = miou
            best_mt = thresh
            fmpred = y_pred

    return best_miou, [-best_mt, best_mt], fmpred


def compute_auc_mc(diffz, y):
    classes = ["No change", "Addition", "Deletion"]
    scores = {}
    for value in classes:
        y_tmp = y.copy()
        diff_tmp = diffz.copy()

        if value == "No change":
            diff_tmp = np.abs(diff_tmp)
            y_tmp[y_tmp > 0] = 1
            scores[value] = roc_auc_score(y_tmp, diff_tmp)
        elif value == "Addition":
            y_tmp[y_tmp == 2] = 0
            scores[value] = roc_auc_score(y_tmp, diff_tmp)
        elif value == "Deletion":
            y_tmp[y_tmp == 1] = 0
            y_tmp[y_tmp == 2] = 1
            diff_tmp = -diff_tmp
            scores[value] = roc_auc_score(y_tmp, diff_tmp)
    return scores
