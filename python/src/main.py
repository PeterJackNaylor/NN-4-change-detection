import pandas as pd
import numpy as np
from function_estimation import (
    estimate_density,
    predict_loop,
)
from parser import parser_f
from data_XYZ import return_dataset, return_dataset_prediction
from architectures import Model
from plotpoint import fig_3d  # plot_surface, plot_tri_grid
import os


def main():
    opt = parser_f()

    time = 0 if opt.csv1 else -1

    model, B, nv, best_score = train_and_test(
        time,
        opt,
    )

    pred_test_save(B, nv, time, model, best_score, opt)


def train_and_test(
    time,
    opt,
    trial=None,
    return_model=True,
):

    train, test, B, nv = return_dataset(
        opt.csv0,
        opt.csv1,
        bs=opt.bs,
        mapping_size=opt.mapping_size,
        fourier=opt.fourier,
        normalize=opt.normalize,
        scale=opt.scale,
        time=not time,
        method=opt.method,
    )

    model = Model(
        train.nn_input,
        arch=opt.architecture,
        activation=opt.activation,
    )
    # model = model.float()
    model = model.cuda()
    hp = {"lr": opt.lr, "epoch": opt.epochs, "wd": opt.wd}
    outputs = estimate_density(
        train,
        test,
        model,
        hp,
        opt.name + ".pth",
        lambda_t=opt.lambda_t,
        method=opt.method,
        trial=trial,
        return_model=return_model,
    )
    if return_model:
        model, best_score = outputs
        return model, B, nv, best_score
    else:
        return outputs


def pred_test_save(
    B,
    nv,
    time,
    model,
    best_score,
    opt,
):

    ds_predict0, xyz0 = return_dataset_prediction(
        opt.csv0,
        opt.csv1,
        bs=opt.bs,
        fourier=opt.fourier,
        B=B,
        nv=nv,
        time=time,
    )

    predictions0 = predict_loop(ds_predict0, model.eval())

    xy = np.array(xyz0.samples.cpu())
    x = xy[:, 0] * nv[0][1] + nv[0][0]
    y = xy[:, 1] * nv[1][1] + nv[1][0]
    f_z0 = np.array(predictions0.cpu())
    f_z0 = f_z0 * nv[2][1] + nv[2][0]

    if opt.csv1:
        ds_predict1, xyz1 = return_dataset_prediction(
            opt.csv0,
            opt.csv1,
            bs=opt.bs,
            fourier=opt.fourier,
            B=B,
            nv=nv,
            time=1,
        )

        predictions1 = predict_loop(ds_predict1, model.eval())
        f_z1 = np.array(predictions1.cpu())
        f_z1 = f_z1 * nv[2][1] + nv[2][0]
    if B:
        B = np.array(B.cpu())
    info = {
        "wd": os.getcwd(),
        "name": opt.name,
        "arch": opt.architecture,
        "score": best_score,
    }
    pd.DataFrame(info, index=[0]).to_csv(opt.name + ".csv")
    if opt.csv1:
        png_snap0 = opt.name + "0.png"
        png_snap1 = opt.name + "1.png"

        fig = fig_3d(x, y, f_z1, f_z1)
        fig.write_image(png_snap1)
        np.savez(
            opt.name + ".npz",
            x=x,
            y=y,
            z0=f_z0,
            z2=f_z1,
            B=B,
            nv=nv,
            score=best_score,
            architecture=opt.architecture,
            activation=opt.activation,
        )
    else:
        png_snap0 = opt.name + ".png"
        np.savez(
            opt.name + ".npz",
            x=x,
            y=y,
            z=f_z0,
            B=B,
            nv=nv,
            score=best_score,
            architecture=opt.architecture,
            activation=opt.activation,
        )

    fig = fig_3d(x, y, f_z0, f_z0)
    fig.write_image(png_snap0)


if __name__ == "__main__":
    main()
