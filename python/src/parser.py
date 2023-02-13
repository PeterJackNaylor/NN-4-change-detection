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
        default="None",
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
