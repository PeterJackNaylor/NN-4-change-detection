import argparse
import yaml


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_yaml(path):
    f = open(path)
    params = yaml.load(f, Loader=yaml.Loader)
    return AttrDict(params)


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
        "--siren",
        action="store_true",
    )
    parser.set_defaults(siren=False)

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
        default="M",
        type=str,
    )
    args = parser.parse_args()

    args.p = read_yaml(args.yaml_file)
    if args.siren:
        args.p.siren = AttrDict(args.p.siren)
    args.p.verbose = args.p.verbose == 1
    return args
