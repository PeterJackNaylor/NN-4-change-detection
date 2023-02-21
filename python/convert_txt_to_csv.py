import argparse
import pandas as pd


def parser_f():

    parser = argparse.ArgumentParser(
        description="From txt to csv",
    )
    parser.add_argument(
        "--txt_file",
        type=str,
    )
    parser.add_argument(
        "--out_file",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    opt = parser_f()
    print("Enter name of txt file: {}".format(opt.txt_file))
    print("writing to txt file: {}".format(opt.txt_file))

    txt = pd.read_csv(opt.txt_file, sep=" ", index_col=None)
    txt.to_csv(opt.out_file, sep=",", index=None)


if __name__ == "__main__":
    main()
