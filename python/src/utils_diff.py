import pandas as pd
import numpy as np
import torch
from function_estimation import predict_loop
from data_XYZ import DataLoader, XYZ_predefinedgrid
from architectures import ReturnModel
from parser import read_yaml


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
    arch = str(npz["architecture"])
    act = str(npz["activation"])

    input_size = 3 if time != -1 else 2

    if four_opt:
        B = torch.tensor(npz["B"]).to("cuda")
    else:
        B = None

    model = ReturnModel(
        input_size,
        arch=arch,
        activation=act,
        B=B,
        fourier=four_opt,
    )
    model.load_state_dict(torch.load(weight))
    return table, model, nv


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

        n0 = weight0.split(".p")[0]
        n1 = weight1.split(".p")[0]

        if tag != 1:
            w0_file = weight0
            w1_file = weight1
            csvfile0 = csv0
            csvfile1 = csv1
            npz_0 = npz0
            npz_1 = npz1
            name0 = n0
            name1 = n1
        else:
            w0_file = weight1
            w1_file = weight0
            csvfile0 = csv1
            csvfile1 = csv0
            npz_0 = npz1
            npz_1 = npz0
            name0 = n1
            name1 = n0
        time = time0 = time1 = -1

        table0, model0, nv0 = load_csv_weight_npz(
            csvfile0, None, w0_file, npz_0, name0, time
        )
        table1, model1, nv1 = load_csv_weight_npz(
            csvfile1, None, w1_file, npz_1, name1, time
        )
        four_opt = name0.split("FOUR=")[1].split("__")[0].split("_")[0]
        four_opt = four_opt == "--fourier"
    else:

        weight = lp.argv[2]

        csvfile0 = lp.argv[3]
        csvfile1 = lp.argv[4]

        npz = lp.argv[5]
        time = 1

        dataname = weight.split("__")[0]
        name = weight.split(".p")[0]

        table, model, nv = load_csv_weight_npz(
            csvfile0,
            csvfile1,
            weight,
            npz,
            name,
            time,
        )
        table0 = table[table["T"] == 0]
        table1 = table[table["T"] == 1]
        model0, model1 = model, model
        nv0, nv1 = nv, nv
        time0 = 0.0
        time1 = 1.0
        four_opt = name.split("FOUR=")[1].split("__")[0].split("_")[0]
        four_opt = four_opt == "--fourier"
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
        four_opt,
        method,
    )


def predict_z(model, nv, grid_indices, normalize, bs=2048, time=0):
    grid = grid_indices.copy()
    iterator = XYZ_predefinedgrid(
        grid,
        nv,
        normalize=normalize,
        time=time,
    )
    loader = DataLoader(
        iterator,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    # model.eval()
    model = model.cuda()
    z = predict_loop(loader, model.eval())
    z = np.array(z.cpu())
    z = z * nv[2][1] + nv[2][0]
    return z
