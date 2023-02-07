import torch
import torch.nn as nn
import numpy as np


def LinearBlock(n_in, n_out, activation='relu'):
    # do not work with ModuleList here either.
    linear = nn.Linear(n_in, n_out)
    if activation == 'relu':
        activ = nn.ReLU()
    elif activation == 'tanh':
        activ = nn.Tanh()
    else:
        raise ValueError('Define activation function')
    block = nn.Sequential(linear, activ)
    return block

def continuous_diff(x, model):
        torch.set_grad_enabled(True)
        x.requires_grad_(True)
        y = model(x)
        dy_dx = torch.autograd.grad(
                    y, x, torch.ones_like(y), 
                    retain_graph=True, create_graph=True,)[0]
        return dy_dx

class MLP(nn.Module):
    
    def __init__(self, in_dim, out_dim, hidden_num, hidden_dim, inner_act_fun):
        super(MLP, self).__init__()

        dim_layers = [in_dim] + hidden_num * [hidden_dim] + [out_dim]
        assert hidden_dim > hidden_num

        layers = []
        num_layers = len(dim_layers)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(LinearBlock(dim_layers[l], dim_layers[l+1], activation=inner_act_fun))
            
        blocks.append(nn.Linear(dim_layers[-2], dim_layers[-1]))
        self.network = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.network(x)


class RFF(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_num, hidden_dim, hidden_act_fun,
                    feature_dim, feature_scale):
        super(RFF, self).__init__()

        self.net = MLP(2*feature_dim, out_dim, hidden_num, hidden_dim, inner_act_fun=hidden_act_fun)

        self.B = torch.nn.Parameter(torch.randn((in_dim, feature_dim)) * 2 * np.pi * feature_scale, requires_grad=False)

    def positional_encoding(self, x):
        return torch.cat((torch.cos(x @ self.B), torch.sin(x @ self.B)), dim=-1)

    def forward(self, xin):
        emb_x = self.positional_encoding(xin)
        return self.net(emb_x)


# create the GON network (a SIREN as in https://vsitzmann.github.io/siren/)
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=[30, 30, 180], is_first=False, is_last=False):
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
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-3,3)

    def forward(self, x, cond_freq=None, cond_phase=None):
        x = self.linear(x)
        if not cond_freq is None:
            freq = cond_freq #.unsqueeze(1).expand_as(x)
            x = freq * x
        if not cond_phase is None:
            phase_shift = cond_phase #unsqueeze(1).expand_as(x)
            x = x + phase_shift
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_num, hidden_dim, feature_scales):
        super(SIREN, self).__init__()
        self.hidden_dim = hidden_dim
        model_dim = [in_dim] + hidden_num * [hidden_dim] + [out_dim]
        assert len(feature_scales) == in_dim
        first_layer = SirenLayer(model_dim[0], model_dim[1], w0=feature_scales, is_first=True)
        other_layers = []
        for dim0, dim1 in zip(model_dim[1:-2], model_dim[2:-1]):
            other_layers.append(SirenLayer(dim0, dim1))
            # other_layers.append(nn.LayerNorm(dim1))
        final_layer = SirenLayer(model_dim[-2], model_dim[-1], is_last=True)
        self.model = nn.Sequential(first_layer, *other_layers, final_layer)

    def forward(self, xin):
        return self.model(xin)

class GaborFilter(nn.Module):
    def __init__(self, in_dim, out_dim, weight_scales, alpha=2, beta=1.0):
        super(GaborFilter, self).__init__()

        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim)

        # Init weights
        for k in range(in_dim):
            self.linear.weight.data[:,k] *=  weight_scales[k] * torch.sqrt(self.gamma)
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        norm = (x ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(x))


class FourierFilter(nn.Module):
    def __init__(self, in_dim, out_dim, weight_scales, bias=True):

        super(FourierFilter, self).__init__()

        assert len(weight_scales) == in_dim
        
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=bias)
        for k in range(in_dim): 
            self.linear.weight.data[:,k] *= weight_scales[k]

        self.linear.bias.data.uniform_(-np.pi, np.pi)
        
    def forward(self, x):
        return torch.sin(self.linear(x))


def mfn_weights_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input), np.sqrt(6/num_input))


class MFN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_num, hidden_dim, feature_scales=[128.], filter='Fourier'):
        super(MFN, self).__init__()

        if filter == 'Fourier':
            filter_fun = FourierFilter
        elif filter == 'Gabor':
            filter_fun = GaborFilter
            
        quantization_interval = 2 * np.pi
        assert len(feature_scales) == in_dim
        input_scales = [round((np.pi * freq / (hidden_num + 1))
                / quantization_interval) * quantization_interval for freq in feature_scales]

        self.K = hidden_num
        self.filters = nn.ModuleList(
            [filter_fun(in_dim, hidden_dim, input_scales) for _ in range(self.K)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(self.K - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])
        self.linear.apply(mfn_weights_init)

    def forward(self, x):
        # Recursion - Equation 3
        zi = self.filters[0](x)  # Eq 3.a
        for i in range(self.K - 1):
            zi = self.linear[i](zi) * self.filters[i + 1](x)  # Eq 3.b

        x = self.linear[-1](zi)  # Eq 3.c
        return x
