import pandas as pd
import numpy as np
import argparse

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

# Fourier feature mapping
def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2.*np.pi*x) @ B.T
        return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

class XYZ(Dataset):

    def __init__(self, csv_file0, csv_file1, train=False, train_fraction=0.8, fourier=False, seed=42, predict=False, scale=1.0, mapping_size=256, B=None, normalize="mean", nv=None, time=False, gradient_regul=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.table = pd.read_csv(csv_file0)[['X', 'Y', 'Z']]
        self.table["T"] = 0
        if csv_file1:
            table1 = pd.read_csv(csv_file1)[['X', 'Y', 'Z']]
            table1["T"] = 1
            self.table = pd.concat([self.table, table1], axis=0).reset_index(drop=True)
        max_coord = (self.table.X.max(), self.table.Y.max(), self.table.Z.max())
        min_coord = (self.table.X.min(), self.table.Y.min(), self.table.Z.min())
        n = self.table.shape[0]
        if not predict:
            idx = np.arange(n)
            np.random.seed(seed)
            np.random.shuffle(idx)
            n0 = int(n * train_fraction)
            idx = idx[:n0] if train else idx[n0:]
            self.table = self.table.loc[idx]
        self.table = self.table.reset_index(drop=True)
        if normalize:
            if nv is None:
                nv_l = []
                for i, var in enumerate(self.table.columns[:-1]):
                    if normalize == "mean":
                        m, s = self.table[var].mean(), self.table[var].std()
                    elif normalize == "one_minus":
                        m =  (max_coord[i] + min_coord[i]) / 2
                        s =  (max_coord[i] - min_coord[i]) / 2
                    nv_l.append((m, s))
                self.nv = nv_l
            else:
                self.nv = nv
            for i, var in enumerate(self.table.columns[:-1]):
                self.table[var] = (self.table[var] - self.nv[i][0]) / self.nv[i][1]
        self.grad_regul = gradient_regul
        self.fourier = fourier
        self.time = time
        self.input_size = 3 if self.time else 2
        if self.fourier:
            if B is not None:
                self.B = B
            else:
                self.B = np.random.normal(size=(mapping_size, self.input_size)) * scale
                self.input_size = mapping_size * 2

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.table.loc[idx, 'X']
        y = self.table.loc[idx, 'Y']
        if self.time:
            t = self.table.loc[idx, 'T']
            sample = np.array([x, y, t])
            if self.grad_regul:
                sample_t = np.array([x, y, 1-t])
        else:
            sample = np.array([x, y])

        target = self.table.loc[idx, 'Z']

        if self.fourier:
            sample = self.fourier_transform(sample)
            if self.grad_regul:
                sample_t = self.fourier_transform(sample_t)
                sample_t = torch.tensor(sample_t).float()
        sample = torch.tensor(sample).float()
        target = target.astype(np.float32)
        if self.grad_regul:
            return sample, sample_t, target
        else:
            return sample, target

    def fourier_transform(self, sample):
        t_sample = input_mapping(sample, self.B)
        return t_sample


class XYZ_predict(Dataset):

    def __init__(self, csv_file0, csv_file1=None, fourier=False, B=None, normalize="mean", nv=None, time=-1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.table = pd.read_csv(csv_file0)[['X', 'Y', 'Z']]
        self.table["T"] = 0
        n = self.table.shape[0]

        if csv_file1:
            table1 = pd.read_csv(csv_file1)[['X', 'Y', 'Z']]
            table1["T"] = 1
            self.table = pd.concat([self.table, table1], axis=0)

        self.time = time
        self.table = self.table.reset_index(drop=True)
        self.fourier = fourier
        if self.fourier:
            self.B = B


        xmax = int(self.table.X.max())
        xmin = int(self.table.X.min())

        ymax = int(self.table.Y.max())
        ymin = int(self.table.Y.min())

        xx, yy = np.meshgrid(np.arange(xmin, xmax, 2), np.arange(ymin, ymax, 2))
        xx = xx.astype(float)
        yy = yy.astype(float)
        self.indices = np.vstack([xx.ravel(), yy.ravel()]).T
        if normalize:
            for i in range(2):
                self.indices[:, i] = (self.indices[:, i] - nv[i][0]) / nv[i][1]
        self.xmax, self.ymax, self.xmin, self.ymin = xmax, ymax, xmin, ymin

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x, y = self.indices[idx]
        if self.time == -1:
            sample = np.array([x, y])
        else:
            sample = np.array([x, y, self.time])
        if self.fourier:
            sample = self.fourier_transform(sample)

        sample = torch.tensor(sample).float()
        return sample

    def fourier_transform(self, sample):
        t_sample = input_mapping(sample, self.B)
        return t_sample

def return_dataset_prediction(csv0, csv1, bs=2048, workers=8, fourier=False, B=None, normalize="mean", nv=None, time=0):
    xyz = XYZ_predict(csv0, csv1, fourier=fourier, B=B, nv=nv, time=time)
    loader = DataLoader(
        xyz,
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, xyz

def return_dataset(csv0, csv1=None, bs=2048, workers=8, mapping_size=256, fourier=False, normalize="mean", scale=1.0, time=False, gradient_regul=False):
    xyz_train = XYZ(csv0, csv1, train=True, mapping_size=mapping_size, fourier=fourier, scale=scale, normalize=normalize, time=time, gradient_regul=gradient_regul)
    nv = xyz_train.nv
    B = xyz_train.B if fourier else None
    xyz_test = XYZ(csv0, csv1, train=False, mapping_size=mapping_size, fourier=fourier, B=B, normalize=normalize, nv=nv, time=time)
    while xyz_train.table.shape[0] < bs:
        bs = bs // 2
    train_loader = DataLoader(
        xyz_train,
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    while xyz_test.table.shape[0] < bs:
        bs = bs // 2
    test_loader = DataLoader(
        xyz_test,
        batch_size=bs,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.input_size = xyz_train.input_size
    return train_loader, test_loader, B, nv


def predict_loop(dataloader, model):
    
    preds = []
    with torch.no_grad():
        for X in dataloader:
            X = X.cuda(non_blocking=True)
            pred = model(X)
            if X.shape[0] == 1:
                pred = torch.Tensor([pred]).cuda()
            preds.append(pred)
    preds = torch.cat(preds)
    return preds

def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, z in dataloader:
            X, z = X.cuda(non_blocking=True), z.cuda(non_blocking=True)
            pred = model(X)
            test_loss = test_loss + loss_fn(pred, z).item()

    test_loss /= num_batches
    
    print(f"\n Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

def estimate_density(dataset, dataset_test, model, hp, name, lambda_t=1.0, gradient_regul=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
    loss_fn = nn.MSELoss()
    if gradient_regul:
        loss_fn_t = nn.L1Loss()
    model.train()
    best_test_score = np.inf
    best_epoch = 0
    for epoch in trange(1, hp["epoch"]+1):

        running_loss, total_num, train_bar = 0.0, 0, tqdm(dataset)
        for data_tuple in train_bar:
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                if gradient_regul:
                    inp, inp_t, target = data_tuple
                else:
                    inp, target = data_tuple
                    
                inp, target = inp.cuda(non_blocking=True), target.cuda(non_blocking=True)
                target_pred = model(inp)
                loss = loss_fn(target_pred, target)

                if gradient_regul:
                    inp_t = inp_t.cuda(non_blocking=True)
                    target_pred_t = model(inp_t)
                    loss += lambda_t * loss_fn_t(target_pred, target_pred_t)

                loss.backward()
                optimizer.step()

                # opti.step()
                running_loss = running_loss + loss.item()
                total_num = total_num + 1
                text = "Train Epoch [{}/{}] Loss: {:.4f}".format(
                    epoch, hp["epoch"], running_loss / total_num
                )
                train_bar.set_description(text)

        if epoch % 10 == 0:
            test_score = test_loop(dataset_test, model, loss_fn)
            if test_score < best_test_score:
                best_test_score = test_score
                best_epoch = epoch
                print(f"best model is now from epoch {epoch}")
                torch.save(
                    model.state_dict(),
                    name
                )
            elif epoch - best_epoch > 20:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 10

    model.load_state_dict(torch.load(name))
    return model, best_test_score

class Model(nn.Module):
    def __init__(self, input_size, p=0.5, activation="tanh"):
        super(Model, self).__init__()
        if activation == "tanh":
            act = nn.Tanh
        elif activation == "relu":
            act = nn.ReLU
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, 256),
            # nn.Dropout(p=p),
            nn.BatchNorm1d(256),
            act(),
            nn.Linear(256, 128),
            # nn.Dropout(p=p),
            nn.BatchNorm1d(128),
            act(),
            nn.Linear(128, 64),
            # nn.Dropout(p=p),
            nn.BatchNorm1d(64),
            act(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        regression = self.linear_stack(x)
        regression = torch.squeeze(regression)
        return regression

def parser_f():

    parser = argparse.ArgumentParser(
        description="Train supervised NN on cell",
    )
    parser.add_argument(
        "--csv0",
        type=str,
    )
    parser.add_argument(
        "--csv1",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--bs",
        default=2048,
        type=int,
        help="Number of images in each mini-batch",
    )
    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        help="Number of workers",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of sweeps over the dataset to train",
    )
    parser.add_argument(
        "--mapping_size",
        default=64,
        type=int,
        help="Number of features to project the vector v",
    )
    parser.add_argument(
        "--fourier",
        action="store_true",
    )
    parser.set_defaults(fourier=False)

    parser.add_argument(
        "--scale",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--normalize",
        default="mean",
        type=str,
    )
    parser.add_argument(
        "--name",
        default="last",
        type=str,
    )
    parser.add_argument(
        "--activation",
        default="tanh",
        type=str,
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "--wd",
        default=0.0005,
        type=float,
    )
    parser.add_argument(
        "--lambda_t",
        default=0.,
        type=float,
    )
    args = parser.parse_args()
    args.gradient_regul = args.lambda_t != 0
    return args

