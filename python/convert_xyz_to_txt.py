import argparse
from tqdm import tqdm
import os


def parser_f():

    parser = argparse.ArgumentParser(
        description="From xyz to txt",
    )
    parser.add_argument(
        "--xyz_file",
        type=str,
    )
    parser.add_argument(
        "--out_txt",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    opt = parser_f()
    print("Enter name of XYZ file: {}".format(opt.xyz_file))
    print("writing to txt file: {}".format(opt.out_txt))

    xyz = open(opt.xyz_file)
    if os.path.exists(opt.out_txt):
        os.remove(opt.out_txt)
    txt = open(opt.out_txt, "w")
    for line in tqdm(xyz):
        txt.write(line.replace(" ", ","))
    xyz.close()
    txt.close()


if __name__ == "__main__":
    main()
