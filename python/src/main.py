import numpy as np
from function_estimation import (
    estimate_density,
    predict_loop,
)
from parser import parser_f, AttrDict
from data_XYZ import return_dataset, return_dataset_prediction
from architectures import ReturnModel, gen_b
from plotpoint import scatter2d  # plot_surface, plot_tri_grid


def main():
    opt = parser_f()
    model_hp = AttrDict()

    model_hp.fourier = opt.fourier
    model_hp.siren = opt.siren
    model_hp.verbose = opt.p.verbose
    model_hp.epochs = opt.p.epochs

    time = 0 if opt.csv1 else -1
    model_hp.bs = 1024 * 16
    model_hp.mapping_size = 512
    model_hp.scale = 4
    # model_hp.architecture = "skip-5"  # "Vlarge"
    # model_hp.activation = "relu"
    model_hp.lr = 0.0001
    model_hp.wd = 0.00005
    opt.method = "M+L1TD"
    opt.p.norm = "one_minus"

    model_hp.L1_time_discrete = "L1TD" in opt.method
    model_hp.L1_time_gradient = "L1TG" in opt.method
    model_hp.tvn = "TVN" in opt.method
    if model_hp.L1_time_discrete:
        model_hp.lambda_discrete = 0.08
    if model_hp.L1_time_gradient:
        model_hp.lambda_gradient_time = 0.2
    if model_hp.tvn:
        model_hp.lambda_tvn = 0.005
        model_hp.loss_tvn = "l1"

    model, model_hp = train_and_test(time, opt, model_hp)

    pred_test_save(model, model_hp, time, opt)


def train_and_test(time, opt, model_hp, trial=None, return_model=True):

    train, test, nv = return_dataset(
        opt.csv0,
        opt.csv1,
        bs=model_hp.bs,
        normalize=opt.p.norm,
        time=not time,
        method=opt.method,
    )
    model_hp.nv = nv
    if model_hp.fourier:
        model_hp.B = gen_b(
            model_hp.mapping_size,
            model_hp.scale,
            train.input_size,
        )

    model = ReturnModel(
        train.input_size,
        arch=model_hp.architecture,
        args=model_hp,
    )

    model = model.cuda()

    outputs = estimate_density(
        train,
        test,
        model,
        model_hp,
        opt.name,
        trial=trial,
        return_model=return_model,
    )
    if return_model:
        model, best_score = outputs
        model_hp.best_score = best_score
        return model, model_hp
    else:
        model_hp.best_score = outputs
        return model_hp


def pred_test_save(
    model,
    model_hp,
    time,
    opt,
):
    nv = model_hp.nv
    xyz0 = return_dataset_prediction(
        opt.csv0,
        opt.csv1,
        nv=model_hp.nv,
        time=time,
    )

    predictions0 = predict_loop(xyz0, model_hp.bs, model.eval())

    xy = np.array(xyz0.samples.cpu())
    x = xy[:, 0] * nv[0][1] + nv[0][0]
    y = xy[:, 1] * nv[1][1] + nv[1][0]
    f_z0 = np.array(predictions0.cpu())
    f_z0 = f_z0 * nv[2][1] + nv[2][0]

    if opt.csv1:
        xyz1 = return_dataset_prediction(
            opt.csv0,
            opt.csv1,
            nv=nv,
            time=1,
        )

        predictions1 = predict_loop(xyz1, model_hp.bs, model.eval())
        f_z1 = np.array(predictions1.cpu())
        f_z1 = f_z1 * nv[2][1] + nv[2][0]
    if "B" in model_hp.keys():
        model_hp.B = np.array(model_hp.B.cpu())

    np.savez(
        opt.name + ".npz",
        **model_hp,
    )

    fig = scatter2d(x, y, f_z0)
    png_snap0 = opt.name + "0.png"
    fig.write_image(png_snap0)

    if opt.csv1:
        png_snap1 = opt.name + "1.png"
        fig = scatter2d(x, y, f_z1)
        fig.write_image(png_snap1)


if __name__ == "__main__":
    main()
