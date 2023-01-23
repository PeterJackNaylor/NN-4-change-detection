import sys

import numpy as np
from plyfile import PlyData, PlyElement


def ply_to_npy(f):
    with open(f, 'rb') as f:
        plydata = PlyData.read(f)

    data = plydata['Urb3DSimul']
    x = data.data['x']
    y = data.data['y']
    z = data.data['z']
    label = data.data['label_ch']

    table = np.vstack([x, y, z, label]).T
    return table

def save(fname, table):
    with open(fname.replace(".ply", ".txt"), "ab") as f:
        f.write(b"X,Y,Z,label_ch\n")
        np.savetxt(f, table, delimiter=',', fmt="%.4f")


def main():
    file0 = sys.argv[1]
    file1 = sys.argv[2]

    table0 = ply_to_npy(file0)
    table1 = ply_to_npy(file1)
    
    n0 = table0.shape[0]
    n1 = table1.shape[0]

    # Shift coordinates
    xmed = np.median(np.concatenate([table0[:,0], table1[:,0]], axis=0))
    ymed = np.median(np.concatenate([table0[:,1], table1[:,1]], axis=0))

    table0[:,0] = table0[:,0] - xmed
    table1[:,0] = table1[:,0] - xmed

    table0[:,1] = table0[:,1] - ymed
    table1[:,1] = table1[:,1] - ymed

    save(file0, table0)
    save(file1, table1)

if __name__ == "__main__":
    main()
