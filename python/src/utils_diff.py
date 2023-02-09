import pandas as pd
import numpy as np
import torch
from function_estimation import predict_loop
from data_XYZ import DataLoader, XYZ_predefinedgrid
from math import ceil, floor
from architectures import Model


def load_csv_weight_npz(csv_file0, csv_file1, weight, npz, name, time=-1):

    table = pd.read_csv(csv_file0)[["X", "Y", "Z", "label_ch"]]
    table.columns = ["X", "Y", "Z", "label"]
    table["T"] = 0
    if csv_file1:
        table1 = pd.read_csv(csv_file1)[["X", "Y", "Z", "label_ch"]]
        table1.columns = ["X", "Y", "Z", "label"]
        table1["T"] = 1
        table = pd.concat([table, table1], axis=0)
    table = table.reset_index(drop=True)

    four_opt = name.split("FOUR=")[1].split("__")[0].split("_")[0]
    four_opt = four_opt == "--fourier"
    npz = np.load(npz)
    nv = npz["nv"]
    arch = npz["architecture"]
    act = npz["activation"]

    input_size = 3 if time != -1 else 2

    if four_opt:
        B = torch.tensor(npz["B"]).to("cuda")
    else:
        B = None

    model = Model(input_size, arch=arch, activation=act, B=B, fourier=four_opt)
    model.load_state_dict(torch.load(weight))
    return table, model, nv


def define_grid(table0, table1, step=2):
    xmin = min(floor(table0.X.min()), floor(table1.X.min()))
    xmax = max(ceil(table0.X.max()), floor(table1.X.max()))
    ymin = min(floor(table0.Y.min()), floor(table1.Y.min()))
    ymax = max(ceil(table0.Y.max()), floor(table1.Y.max()))
    xr, yr = np.arange(xmin, xmax, step), np.arange(ymin, ymax, step)
    xx, yy = np.meshgrid(xr, yr)

    xx = xx.astype(float)
    yy = yy.astype(float)
    indices = np.vstack([xx.ravel(), yy.ravel()]).T
    return indices


def predict_z(model, nv, grid_indices, normalize, bs=2048, time=0):
    iterator = XYZ_predefinedgrid(
        grid_indices,
        nv,
        normalize=normalize,
        time=time,
    )
    loader = DataLoader(
        iterator,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        # pin_memory=True,
        drop_last=False,
    )
    model.eval()
    model = model.cuda()
    z = predict_loop(loader, model)
    z = np.array(z.cpu())
    z = z * nv[2][1] + nv[2][0]
    return z
