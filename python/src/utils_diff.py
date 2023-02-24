import pandas as pd
import numpy as np
import torch
from function_estimation import predict_loop
from data_XYZ import XYZ_predefinedgrid
from architectures import ReturnModel
from parser import read_yaml, AttrDict


def clean_hp(d):
    for k in d.keys():
        if k in ["fourier", "siren", "siren_skip"]:
            d[k] = bool(d[k])
        elif k in []:
            d[k] = float(d[k])
        elif k in [
            "bs",
            "scale",
            "siren_hidden_num",
            "siren_hidden_num",
            "siren_hidden_dim",
        ]:
            d[k] = int(d[k])
        elif k in ["architecture", "activation"]:
            d[k] = str(d[k])
        elif k == "B":
            d.B = torch.tensor(d.B).to("cuda")
    return d


def load_tables(csv0, csv1):
    table = pd.read_csv(csv0)
    label = "label_ch" in table.columns
    if label:
        columns = ["X", "Y", "Z", "label_ch"]
    else:
        columns = ["X", "Y", "Z"]
    table = table[columns]
    if label:
        table.columns = ["X", "Y", "Z", "label"]
    table["T"] = 0
    if csv1:
        table1 = pd.read_csv(csv1)[columns]
        if label:
            table1.columns = ["X", "Y", "Z", "label"]
        table1["T"] = 1
        table = pd.concat([table, table1], axis=0)
    table = table.reset_index(drop=True)
    return table


def load_csv_weight_npz(csv_file0, csv_file1, weight, npz, time=-1):

    table = load_tables(csv_file0, csv_file1)

    npz = np.load(npz)
    hp = AttrDict(npz)
    hp = clean_hp(hp)

    input_size = 3 if time != -1 else 2

    model = ReturnModel(
        input_size,
        arch=str(hp.architecture),
        args=hp,
    )
    model.load_state_dict(torch.load(weight))
    return table, model, hp


def load_data(lp):
    yaml_file = read_yaml(lp.argv[-1])
    normalize = yaml_file.norm
    method = lp.argv[1]
    if "double" in method:
        weight0 = lp.argv[2]
        weight1 = lp.argv[3]

        csv0 = lp.argv[4]
        csv1 = lp.argv[5]

        npz0 = lp.argv[6]
        npz1 = lp.argv[7]

        dataname = weight0.split("__")[0][:-1]
        tag = int(weight0.split("__")[0][-1])

        if tag != 1:
            w0_file = weight0
            w1_file = weight1
            csvfile0 = csv0
            csvfile1 = csv1
            npz_0 = npz0
            npz_1 = npz1
        else:
            w0_file = weight1
            w1_file = weight0
            csvfile0 = csv1
            csvfile1 = csv0
            npz_0 = npz1
            npz_1 = npz0
        time = time0 = time1 = -1

        table0, model0, hp0 = load_csv_weight_npz(
            csvfile0,
            None,
            w0_file,
            npz_0,
            time,
        )
        table1, model1, hp1 = load_csv_weight_npz(
            csvfile1,
            None,
            w1_file,
            npz_1,
            time,
        )
        nv0, nv1 = hp0.nv, hp1.nv

        if hp0.fourier:
            fs = "FourierBasis"
        elif hp0.siren:
            fs = "SIREN"
        else:
            fs = "None"
        lambda_t = None
    else:

        weight = lp.argv[2]

        csvfile0 = lp.argv[3]
        csvfile1 = lp.argv[4]

        npz = lp.argv[5]
        time = 1

        dataname = weight.split("__")[0]

        table, model, hp = load_csv_weight_npz(
            csvfile0,
            csvfile1,
            weight,
            npz,
            time,
        )
        table0 = table[table["T"] == 0]
        table1 = table[table["T"] == 1]
        model0, model1 = model, model
        nv0, nv1 = hp.nv, hp.nv
        time0 = 0.0
        time1 = 1.0
        if hp.fourier:
            fs = "FourierBasis"
        elif hp.siren:
            fs = "SIREN"
        else:
            fs = "None"
        if "L1TD" in method:
            lambda_t = float(np.load(npz)["lambda_discrete"])
        elif "TVN" in method:
            lambda_t = float(np.load(npz)["lambda_tvn"])
        else:
            lambda_t = None

    return (
        table0,
        table1,
        model0,
        model1,
        nv0,
        nv1,
        time0,
        time1,
        dataname,
        normalize,
        fs,
        method,
        lambda_t,
    )


def predict_z(model, nv, grid_indices, normalize, bs=2048, time=0):
    grid = grid_indices.copy()
    iterator = XYZ_predefinedgrid(
        grid,
        nv,
        normalize=normalize,
        time=time,
    )
    model = model.cuda()
    z = predict_loop(iterator, bs, model.eval())
    z = np.array(z.cpu())
    z = z * nv[2][1] + nv[2][0]
    return z
