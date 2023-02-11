import argparse
import yaml


def read_yaml(file):
    f = open(file)
    param_grid = yaml.load(f, Loader=yaml.Loader)
    res = {
        "norm": param_grid["norm"],
        "lr": param_grid["lr"],
        "wd": param_grid["wd"],
        "bs": param_grid["bs"],
        "scale": param_grid["scale"],
        "mp_size": param_grid["mapping_size"],
        "lambda_t": param_grid["lambda_t"],
        "act": param_grid["act"],
        "arch": param_grid["arch"],
    }
    return res


def parser_f():

    parser = argparse.ArgumentParser(
        description="Train supervised NN on cell",
    )
    parser.add_argument(
        "--csv0",
        type=str,
    )
    parser.add_argument(
        "--csv1",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--fourier",
        action="store_true",
    )
    parser.set_defaults(fourier=False)

    parser.add_argument(
        "--name",
        default="last",
        type=str,
    )
    parser.add_argument(
        "--yaml_file",
        default="paper.yaml",
        type=str,
    )
    parser.add_argument(
        "--method",
        default="method",
        type=str,
    )
    args = parser.parse_args()

    f = open(args.yaml_file)
    params = yaml.load(f, Loader=yaml.Loader)
    args.epochs = params["epochs"]
    args.trials = params["trials"]
    args.normalize = params["norm"]
    args.verbose = params["verbose"] == 1
    args.workers = 0
    return args

    # parser.add_argument(
    #     "--activation",
    #     default="tanh",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--bs",
    #     default=2048,
    #     type=int,
    #     help="Number of images in each mini-batch",
    # )
    # parser.add_argument(
    #     "--workers",
    #     default=1,
    #     type=int,
    #     help="Number of workers",
    # )
    # parser.add_argument(
    #     "--epochs",
    #     default=100,
    #     type=int,
    #     help="Number of sweeps over the dataset to train",
    # )
    # parser.add_argument(
    #     "--mapping_size",
    #     default=64,
    #     type=int,
    #     help="Number of features to project the vector v",
    # )
    # parser.add_argument(
    #     "--trials",
    #     default=32,
    #     type=int,
    #     help="Number of trials for optuna",
    # )
    # parser.add_argument(
    #     "--lr",
    #     default=0.01,
    #     type=float,
    # )
    # parser.add_argument(
    #     "--wd",
    #     default=0.0005,
    #     type=float,
    # )
    # parser.add_argument(
    #     "--lambda_t",
    #     default=0.0,
    #     type=float,
    # )

    # parser.add_argument(
    #     "--scale",
    #     default=1.0,
    #     type=float,
    # )
    # parser.add_argument(
    #     "--normalize",
    #     default="mean",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--arch",
    #     default="default",
    #     type=str,
    # )
