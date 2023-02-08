import pandas as pd
import numpy as np
import torch
from function_estimation import predict_loop
from data_XYZ import DataLoader, input_mapping
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

    npz = np.load(npz)
    B = torch.tensor(npz["B"]).to("cuda")
    nv = npz["nv"]
    arch = npz["architecture"]
    act = npz["activation"]

    four_opt = name.split("FOUR=")[1].split("__")[0].split("_")[0]

    if four_opt == "--fourier":
        mappingsize = B.shape[0]
        input_size = mappingsize * 2
    else:
        input_size = 3 if time != -1 else 2

    model = Model(input_size, arch=arch, activation=act)
    model.load_state_dict(torch.load(weight))
    return table, model, B, nv, four_opt


class indice_iterator:
    def __init__(self, samples, B, nv, time, fourier):
        fourier = fourier == "--fourier"
        self.B = B
        self.time = time
        for i in range(2):
            samples[:, i] = (samples[:, i] - nv[i][0]) / nv[i][1]
        if self.time != -1:
            T = np.zeros_like(samples[:, 0]) + self.time
            samples = np.array([samples[:, 0], samples[:, 1], T]).T
        self.samples = torch.tensor(samples).float()
        self.samples = self.samples.to("cuda")
        if fourier:
            self.samples = input_mapping(self.samples, self.B)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx]

    def fourier_transform(self, sample):
        t_sample = input_mapping(sample, self.B)
        return t_sample


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


def predict_z(model, B, nv, grid_indices, fourier, bs=2048, time=0):
    iterator = indice_iterator(grid_indices.copy(), B, nv, time, fourier)
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
