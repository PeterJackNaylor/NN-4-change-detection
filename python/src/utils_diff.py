import pandas as pd
import numpy as np
import torch
from function_estimation import Model, input_mapping, predict_loop, DataLoader, predict_loop
from math import ceil, floor

def load_csv_weight_npz(csv_file0, csv_file1, weight, npz, name, time=-1):

    table = pd.read_csv(csv_file0)[['X', 'Y', 'Z', 'label_ch']]
    table.columns = ['X', 'Y', 'Z', 'label']
    table['T'] = 0
    if csv_file1:
        table1 = pd.read_csv(csv_file1)[['X', 'Y', 'Z', 'label_ch']]
        table1.columns = ['X', 'Y', 'Z', 'label']
        table1['T'] = 1
        table = pd.concat([table, table1], axis=0)
    table = table.reset_index(drop=True)

    four_opt = name.split("FOUR=")[1].split("__")[0]

    mappingsize = int(name.split("MAPPINGSIZE=")[1].split("_")[0])
    if four_opt == "--fourier":
        input_size = mappingsize * 2
    else:
        input_size = 3 if time != -1 else 2

    act = name.split("ACT=")[1].split("_")[0]

    model = Model(input_size, p=0.5, activation=act)
    model.load_state_dict(torch.load(weight))
    npz = np.load(npz)
    B = npz['B']
    nv = npz['nv']
    return table, model, B, nv

class indice_iterator:
    def __init__(self, indices, B, nv, time):
        self.indices = indices
        self.B = B
        self.time = time
        for i in range(2):
            self.indices[:, i] = (self.indices[:, i] - nv[i][0]) / nv[i][1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x, y = self.indices[idx]
        if self.time != -1:
            sample = np.array([x, y, self.time])
        else:
            sample = np.array([x, y])
        sample = self.fourier_transform(sample)

        sample = torch.tensor(sample).float()
        return sample

    def fourier_transform(self, sample):
        t_sample = input_mapping(sample, self.B)
        return t_sample

def define_grid(table0, table1, step=2):
    xmin = min(floor(table0.X.min()), floor(table1.X.min()))
    xmax = max(ceil(table0.X.max()), floor(table1.X.max()))
    ymin = min(floor(table0.Y.min()), floor(table1.Y.min()))
    ymax = max(ceil(table0.Y.max()), floor(table1.Y.max()))

    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymin, ymax, step))
    xx = xx.astype(float)
    yy = yy.astype(float)
    indices = np.vstack([xx.ravel(), yy.ravel()]).T
    return indices

def predict_z(model, B, nv, grid_indices, bs=2048, workers=1, time=0):
    iterator = indice_iterator(grid_indices.copy(), B, nv, time)
    loader = DataLoader(
                iterator,
                batch_size=bs,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
                drop_last=False,
        )
    model.eval()
    model = model.cuda()
    z = predict_loop(loader, model)
    z = np.array(z.cpu())
    z = z * nv[2][1] + nv[2][0]
    return z
