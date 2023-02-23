import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class XYZ(Dataset):
    def __init__(
        self,
        csv_file0,
        csv_file1,
        train_fold=False,
        train_fraction=0.8,
        seed=42,
        pred_type="table_predictions",
        step_grid=2.0,
        nv=None,
        normalize="mean",
        time=False,
        method="M",
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.time = time
        self.need_inverse_time = "L1TD" in method or "L1TG" in method
        self.need_target = not pred_type == "grid_predictions"
        self.nv = nv
        input_size = 3 if self.time else 2
        self.input_size = input_size
        self.step_grid = step_grid

        table = self.read_table(csv_file0, 0)

        if csv_file1:
            table1 = self.read_table(csv_file1, 1)
            table = pd.concat([table, table1], axis=0)
        table = table.reset_index(drop=True)

        if pred_type == "table_predictions":
            self.setup_table_sample(table)
            self.split_train(seed, train_fraction, train_fold)

        elif pred_type == "grid_predictions":
            self.setup_uniform_grid(table)

        if normalize:
            self.normalize(normalize)

        self.samples = torch.tensor(self.samples).float()
        if self.need_target:
            self.targets = torch.tensor(self.targets)
        self.send_cuda()

    def send_cuda(self):
        self.samples = self.samples.to("cuda")
        if self.need_inverse_time:
            self.samples_t = self.samples_t.to("cuda")
        if self.need_target:
            self.targets = self.targets.to("cuda")

    def read_table(self, path, time):
        table = pd.read_csv(path)[["X", "Y", "Z"]]
        table["T"] = time
        return table

    def split_train(self, seed, train_fraction, train):
        n = self.samples.shape[0]
        idx = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(idx)
        n0 = int(n * train_fraction)
        idx = idx[:n0] if train else idx[n0:]
        self.samples = self.samples[idx]
        self.targets = self.targets[idx]
        if self.time and self.need_inverse_time:
            self.samples_t = self.samples_t[idx]

    def normalize(self, normalize):

        if self.nv is None:
            nv_l = []
            for vect in [self.samples[:, 0], self.samples[:, 1], self.targets]:
                if normalize == "mean":
                    m, s = vect.mean(), vect.std()
                elif normalize == "one_minus":
                    m = (vect.max() + vect.min()) / 2
                    s = (vect.max() - vect.min()) / 2
                elif normalize == "one_zero":
                    m = vect.min()
                    s = vect.max() - vect.min()
                nv_l.append((m, s))
            self.nv = nv_l
        nv = self.nv
        for i in range(2):
            self.samples[:, i] = (self.samples[:, i] - nv[i][0]) / nv[i][1]
            if self.need_inverse_time:
                tmp = (self.samples_t[:, i] - nv[i][0]) / nv[i][1]
                self.samples_t[:, i] = tmp

        if self.need_target:
            self.targets = (self.targets - nv[2][0]) / nv[2][1]

    def setup_table_sample(self, table):
        X = table["X"].values
        Y = table["Y"].values
        self.targets = table["Z"].values.astype(np.float32)
        if self.time:
            T = table["T"].values
            self.samples = np.array([X, Y, T]).T
            if self.need_inverse_time:
                self.samples_t = np.array([X, Y, 1 - T]).T
                self.samples_t = torch.tensor(self.samples_t).float()
        else:
            self.samples = np.array([X, Y]).T

    def setup_uniform_grid(self, table):

        xmax = int(table.X.max())
        xmin = int(table.X.min())

        ymax = int(table.Y.max())
        ymin = int(table.Y.min())

        xx, yy = np.meshgrid(
            np.arange(xmin, xmax, self.step_grid),
            np.arange(ymin, ymax, self.step_grid),
        )
        xx = xx.astype(float)
        yy = yy.astype(float)
        if self.time == -1:
            self.samples = np.vstack([xx.ravel(), yy.ravel()]).T
        else:
            time = np.zeros_like(yy.ravel()) + self.time
            self.samples = np.vstack([xx.ravel(), yy.ravel(), time]).T

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if not self.need_target:
            return sample
        target = self.targets[idx]

        if self.need_inverse_time:
            sample_t = self.samples_t[idx]

        if self.need_inverse_time:
            return sample, sample_t, target
        else:
            return sample, target


# class XYZ_sample(XYZ):
#     def __getitem__(self, idx):
#         return self.samples[idx]


# class XYZ_sample_target(XYZ):
#     def __getitem__(self, idx):
#         return self.samples[idx], self.targets[idx]


# class XYZ_sample_time_target(XYZ):
#     def __getitem__(self, idx):
#         return self.samples[idx], self.samples_t[idx], self.targets[idx]


class XYZ_predefinedgrid(XYZ):
    def __init__(self, grid, nv=None, normalize="mean", time=False):

        self.time = time
        self.need_inverse_time = False
        self.need_target = False
        self.nv = nv
        input_size = 3 if self.time else 2
        self.input_size = input_size

        self.setup_uniform_grid(grid)

        if normalize:
            self.normalize(normalize)

        self.samples = torch.tensor(self.samples).float()
        self.send_cuda()

    def setup_uniform_grid(self, grid):
        if self.time == -1:
            self.samples = grid
        else:
            time = np.zeros_like(grid[:, 0]) + self.time
            self.samples = np.concatenate([grid, time[:, np.newaxis]], axis=1)


def return_dataset_prediction(
    csv0,
    csv1,
    normalize="mean",
    nv=None,
    time=0,
):
    xyz = XYZ(
        csv0,
        csv1,
        pred_type="grid_predictions",
        nv=nv,
        time=time,
        normalize=normalize,
    )
    return xyz


def return_dataset(
    csv0,
    csv1=None,
    bs=2048,
    normalize="mean",
    time=False,
    method="M",
):
    # if "L1TD" in method or "L1TG" in method:
    #     data_xyz = XYZ_sample_time_target
    # else:
    #     data_xyz = XYZ_sample_target
    xyz_train = XYZ(
        csv0,
        csv1,
        train_fold=True,
        train_fraction=0.8,
        seed=42,
        pred_type="table_predictions",
        nv=None,
        normalize=normalize,
        time=time,
        method=method,
    )
    nv = xyz_train.nv

    xyz_test = XYZ(
        csv0,
        csv1,
        train_fold=False,
        train_fraction=0.8,
        seed=42,
        pred_type="table_predictions",
        nv=nv,
        normalize=normalize,
        time=time,
        method="M",
    )
    # n_train = xyz_train.samples.shape[0]
    # bs_train = bs
    # while n_train < bs:
    #     bs_train = bs_train // 2

    # train_loader = DataLoader(
    #     xyz_train,
    #     batch_size=bs,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True,
    # )
    # while xyz_test.samples.shape[0] < bs:
    #     bs = bs // 2
    # test_loader = DataLoader(
    #     xyz_test,
    #     batch_size=bs,
    #     num_workers=0,
    #     drop_last=True,
    # )
    # train_loader.input_size = xyz_train.input_size
    return xyz_train, xyz_test, nv
    # return xyz_train, xyz_test, nv, bs_train
    # return train_loader, test_loader, nv
