import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, input_size, arch="default", activation="tanh"):
        super(Model, self).__init__()
        if activation == "tanh":
            act = nn.Tanh
        elif activation == "relu":
            act = nn.ReLU

        self.mlp_model = gen_arch(arch, act, input_size)

    def forward(self, x):
        regression = self.mlp_model(x)
        regression = torch.squeeze(regression)
        return regression


def gen_arch(arch, act, input_size):
    if arch == "default":
        mod = nn.Sequential(
            nn.Linear(input_size, 256),
            act(),
            nn.Linear(256, 128),
            act(),
            nn.Linear(128, 64),
            act(),
            nn.Linear(64, 1),
        )
    elif arch == "default-BN":
        mod = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            act(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            act(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            act(),
            nn.Linear(64, 1),
        )
    elif arch == "skipMed":
        # need to be sure this type of thing works
        def mod(x):
            x = nn.Linear(input_size, 256)(x)
            x = act()(x)
            x = nn.Linear(256, 128)(x)
            x = act()(x)
            for _ in range(4):
                y = nn.Linear(128, 128)(x)
                y = act()(y)
                x = x + y
            x = nn.Linear(128, 1)(x)
            return x

    elif arch == "Vlarge":
        mod = nn.Sequential(
            nn.Linear(input_size, 1024),
            act(),
            nn.Linear(1024, 512),
            act(),
            nn.Linear(512, 256),
            act(),
            nn.Linear(256, 128),
            act(),
            nn.Linear(128, 64),
            act(),
            nn.Linear(64, 1),
        )
    else:
        raise ValueError("Architecture unkown")
    return mod
