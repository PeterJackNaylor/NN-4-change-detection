import torch.nn as nn
import torch
from numpy import random, float32


def gen_b(mapping, scale, input_size):
    shape = (mapping, input_size)
    B = random.normal(size=shape).astype(float32)
    B = torch.tensor(B).to("cuda")
    return B * scale


def ReturnModel(
    input_size,
    arch="default",
    activation="relu",
    B=None,
    fourier=True,
):
    if arch in ["default", "default-BN", "Vlarge"]:
        mod = Fmodel(
            input_size,
            arch=arch,
            activation=activation,
            B=B,
            fourier=fourier,
        )
    elif "skip" in arch:
        number = int(arch[-1])
        mod = SkipModel(
            input_size,
            activation=activation,
            B=B,
            fourier=fourier,
            number=number,
        )
    return mod


class Fmodel(nn.Module):
    def __init__(
        self,
        input_size,
        arch="default",
        activation="tanh",
        B=None,
        fourier=False,
    ):
        super(Fmodel, self).__init__()
        if activation == "tanh":
            act = nn.Tanh
        elif activation == "relu":
            act = nn.ReLU
        if fourier:
            self.fourier_mapping_setup(B)
            self.first = self.fourier_map
            input_size = self.fourier_size * 2
        else:
            self.first = nn.Identity()
        self.mlp_model = gen_arch(arch, act, input_size)

    def fourier_mapping_setup(self, B):
        n, p = B.shape
        layer = nn.Linear(n, p, bias=False)
        layer.weight = nn.Parameter(B, requires_grad=False)
        layer.requires_grad_(False)
        self.fourier_layer = layer
        self.fourier_size = n

    def fourier_map(self, x):
        x = 2 * torch.pi * self.fourier_layer(x)
        x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1)
        return x

    def forward(self, x):
        x = self.first(x)
        regression = self.mlp_model(x)
        regression = torch.squeeze(regression)
        return regression


class skip_and_down_block(nn.Module):
    def __init__(self, in_channels, out_channels, act, number):
        super(skip_and_down_block, self).__init__()
        layers = []
        for i in range(number):
            layers.append(skipblock(in_channels, act))
        self.backbone = nn.Sequential(*layers)
        if out_channels is not None:
            self.mlp_down = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                act(),
            )
            self.model = nn.Sequential(self.backbone, self.mlp_down)
        else:
            self.model = self.backbone

    def forward(self, x):
        return self.model(x)


class skipblock(nn.Module):
    def __init__(self, in_channels, act):
        super(skipblock, self).__init__()

        self.mlp_skip = nn.Linear(in_channels, in_channels)
        self.act = act()

    def forward(self, x):
        identity = x.clone()
        x = self.mlp_skip(x)
        x = self.act(x)
        x = x + identity
        return x


class SkipModel(Fmodel):
    def __init__(
        self,
        input_size,
        activation="tanh",
        B=None,
        fourier=False,
        number=1,
    ):
        super(SkipModel, self).__init__(
            input_size,
            "default",
            activation,
            B,
            fourier,
        )
        if activation == "tanh":
            self.act = nn.Tanh
        elif activation == "relu":
            self.act = nn.ReLU
        if fourier:
            self.fourier_mapping_setup(B)
            self.first = self.fourier_map
            input_size = self.fourier_size * 2
        else:
            self.first = nn.Identity()

        if number == 1:
            first = 512
            layer_size = [(first, 2), (256, 2), (128, 1), (64, 1)]
        elif number == 2:
            first = 512
            layer_size = [(first, 10), (512, 1), (256, 1), (128, 1), (64, 1)]
        elif number == 3:
            first = 1024
            layer_size = [(first, 2), (512, 2), (256, 2), (128, 2), (64, 2)]
        elif number == 4:
            first = 1024
            layer_size = [(first, 4), (512, 4), (256, 3), (128, 2), (64, 2)]
        elif number == 5:
            first = 256
            layer_size = [(first, 10)]
        elif number == 6:
            first = 256
            layer_size = [(first, 20), (128, 2), (64, 2)]

        layers = []
        self.mlp_size0 = nn.Sequential(
            nn.Linear(input_size, first),
            self.act(),
        )
        for i, (size, number) in enumerate(layer_size):
            if i != (len(layer_size) - 1):
                nnext = layer_size[i + 1][0]
            else:
                nnext = None
            layers.append(skip_and_down_block(size, nnext, self.act, number))
        self.layers = nn.Sequential(*layers)
        self.last = nn.Linear(size, 1)
        # self.model = nn.Sequential(self.mlp_size0, self.layers, self.last)

    def forward(self, x):
        x = self.first(x)
        x = self.mlp_size0(x)
        x = self.layers(x)
        x = self.last(x)
        # x = self.model(x)
        regression = torch.squeeze(x)
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

    # elif arch == "skipMed":
    #     # need to be sure this type of thing works
    #     def mod(x):
    #         x = nn.Linear(input_size, 256)(x)
    #         x = act()(x)
    #         x = nn.Linear(256, 128)(x)
    #         x = act()(x)
    #         for _ in range(4):
    #             y = nn.Linear(128, 128)(x)
    #             y = act()(y)
    #             x = x + y
    #         x = nn.Linear(128, 1)(x)
    #         return x
