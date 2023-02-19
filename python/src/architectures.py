import torch.nn as nn
import torch
from numpy import random, float32, sqrt


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


# create the GON network (a SIREN as in https://vsitzmann.github.io/siren/)
class SirenLayer(nn.Module):
    def __init__(
        self,
        in_f,
        out_f,
        w0=[30, 30, 180],
        is_first=False,
        is_last=False,
    ):
        super().__init__()

        if is_first:
            assert in_f == len(w0)

        self.in_f = in_f
        self.w0 = w0[0]
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        # with torch.no_grad():
        #     if self.is_first:
        #         for k in range(self.in_f):
        #             self.linear.weight.uniform_(-1, 1)
        #             self.linear.weight.data[:,k] *= self.w0[k] / self.in_f
        #     else:
        #         b = np.sqrt(6 / self.in_f)
        #         self.linear.weight.uniform_(-b, b)
        b = 1 / self.in_f if self.is_first else sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-3, 3)

    def forward(self, x, cond_freq=None, cond_phase=None):
        x = self.linear(x)
        if cond_freq is not None:
            freq = cond_freq  # .unsqueeze(1).expand_as(x)
            x = freq * x
        if cond_phase is not None:
            phase_shift = cond_phase  # unsqueeze(1).expand_as(x)
            x = x + phase_shift
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_num,
        hidden_dim,
        feature_scales,
    ):
        super(SIREN, self).__init__()
        self.hidden_dim = hidden_dim
        model_dim = [in_dim] + hidden_num * [hidden_dim] + [out_dim]
        assert len(feature_scales) == in_dim
        first_layer = SirenLayer(
            model_dim[0], model_dim[1], w0=feature_scales, is_first=True
        )
        other_layers = []
        for dim0, dim1 in zip(model_dim[1:-2], model_dim[2:-1]):
            other_layers.append(SirenLayer(dim0, dim1))
            # other_layers.append(nn.LayerNorm(dim1))
        final_layer = SirenLayer(model_dim[-2], model_dim[-1], is_last=True)
        self.model = nn.Sequential(first_layer, *other_layers, final_layer)

    def forward(self, xin):
        return self.model(xin)


class INCANT(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_num, hidden_dim, feature_scales, do_skip
    ):
        super(INCANT, self).__init__()
        self.model = SIREN(
            in_dim,
            out_dim,
            hidden_num,
            hidden_dim,
            feature_scales,
        )
        # self.model = RFF(in_dim, out_dim, hidden_num, hidden_dim,
        # 'relu', feature_dim=1024, feature_scale=feature_scales[0] )
        self.do_skip = do_skip

    # extend SIREN forward
    def forward(self, xin):
        if self.do_skip:
            for i, layer in enumerate(self.model.model):
                if layer.is_first:
                    x = layer(xin)
                elif layer.is_last:
                    out = layer(x)
                else:
                    x = layer(x) + x if i % 2 == 1 else layer(x)
        else:
            out = self.model(xin)
        return torch.tanh(out)

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
