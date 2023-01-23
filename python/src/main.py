
import pandas as pd
import numpy as np
from function_estimation import parser_f, return_dataset, return_dataset_prediction, estimate_density, Model, predict_loop
from plotpoint import fig_3d #plot_surface, plot_tri_grid
import os

def main():
    opt = parser_f()

    bs = opt.bs
    csv0 = opt.csv0
    csv1 = opt.csv1
    time = 0 if csv1 else -1

    workers = opt.workers
    fourier = opt.fourier
    scale = opt.scale
    mapping_size = opt.mapping_size
    normalize = opt.normalize
    weight_file = opt.name + ".pth"
    coord_file = opt.name + ".npz"
    csv_file = opt.name + ".csv"
    activation = opt.activation

    train, test, B, nv = return_dataset(csv0, csv1, bs=bs, workers=workers, mapping_size=mapping_size, fourier=fourier, normalize=normalize, scale=scale, time=not time, gradient_regul=opt.gradient_regul)

    model = Model(train.input_size, activation=activation)
    # model = model.float()
    model = model.cuda()
    hp = {
        "lr": opt.lr, 
        "epoch": opt.epochs,
        "wd": opt.wd
    }
    model, best_score = estimate_density(train, test, model, hp, weight_file, lambda_t=opt.lambda_t, gradient_regul=opt.gradient_regul)

    ds_predict0, xyz0 = return_dataset_prediction(csv0, csv1, bs=bs, workers=workers, fourier=fourier, B=B, nv=nv, time=time)
    predictions0 = predict_loop(ds_predict0, model.eval())

    xy = xyz0.indices
    x = xy[:, 0] * nv[0][1] + nv[0][0]
    y = xy[:, 1] * nv[1][1] + nv[1][0]
    f_z0 = np.array(predictions0.cpu())
    f_z0 = f_z0 * nv[2][1] + nv[2][0]

    if csv1:
        ds_predict1, xyz1 = return_dataset_prediction(csv0, csv1, bs=bs, workers=workers, fourier=fourier, B=B, nv=nv, time=1)

        predictions1 = predict_loop(ds_predict1, model.eval())
        f_z1 = np.array(predictions1.cpu())
        f_z1 = f_z1 * nv[2][1] + nv[2][0]
    
    info = {
        "wd" : os.getcwd(),
        "name": opt.name,
        "score": best_score
    }
    pd.DataFrame(info, index=[0]).to_csv(csv_file)
    if csv1:
        png_snap0 = opt.name + "0.png"
        png_snap1 = opt.name + "1.png"

        fig = fig_3d(x, y, f_z1, f_z1)
        fig.write_image(png_snap1)
        np.savez(coord_file, x=x, y=y, z0=f_z0, z2=f_z1, B=B, nv=nv, score=best_score)
    else:
        png_snap0 = opt.name + ".png"
        np.savez(coord_file, x=x, y=y, z=f_z0, B=B, nv=nv, score=best_score)
    

    fig = fig_3d(x, y, f_z0, f_z0)
    fig.write_image(png_snap0)

    # f_z = f_z.reshape((xyz.xmax-xyz.xmin, xyz.ymax-xyz.ymin), order='F')
    # plot_surface(xyz.xmin, xyz.xmax, xyz.ymin, xyz.ymax, f_z)
if __name__ == "__main__":
    main()