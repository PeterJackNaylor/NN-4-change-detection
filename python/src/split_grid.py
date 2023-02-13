import sys
from numpy import genfromtxt
import numpy as np 

def split_in_2(table, xm, ym, horizontal=True):
    if horizontal:
        table_l = table[table[:,1] <  ym]
        table_r = table[table[:,1] >=  ym]
    else:
        table_l = table[table[:,0] <  xm]
        table_r = table[table[:,0] >= xm]
    return table_l, table_r
    

def recursive_split(t0, t1, x_0, x_1, y_0, y_1, n, current_split=True):
    # current split is for horizontal or vertical split
    if t0.shape[0] < n and t1.shape[0] < n:
        return [(t0, t1, x_0, x_1, y_0, y_1)]
    else:
        # xh = (x_0 + x_1) / 2
        # yh = (y_0 + y_1) / 2
        next_split = not current_split
        
        xh = compute_median(t0, t1, 0)
        yh = compute_median(t0, t1, 1) 
        t0_l, t0_r = split_in_2(t0, xh, yh, current_split)
        t1_l, t1_r = split_in_2(t1, xh, yh, current_split)
        if current_split:
            x_n00 = x_0
            x_n10 = x_1
            y_n00 = y_0
            y_n10 = yh
            x_n01 = x_0
            x_n11 = x_1
            y_n01 = yh
            y_n11 = y_1
        else:
            x_n00 = x_0
            x_n10 = xh
            y_n00 = y_0
            y_n10 = y_1
            x_n01 = xh
            x_n11 = x_1
            y_n01 = y_0
            y_n11 = y_1

    

        out0 = recursive_split(t0_l, t1_l, x_n00, x_n10, y_n00, y_n10, n, next_split)
        out1 = recursive_split(t0_r, t1_r, x_n01, x_n11, y_n01, y_n11, n, next_split)
        
        return out0 + out1

def save(root_f, table, dataname, n_chunk, timestamp):
    fname = root_f.format(n_chunk, dataname, timestamp)
    with open(fname, "ab") as f:
        f.write(b"X,Y,Z,label_ch\n")
        np.savetxt(f, table, delimiter=',', fmt="%.4f")

def compute_median(t0, t1, axis):
    return np.median(np.concatenate([t0[:, axis],t1[:, axis]], axis=0))

def main():

    file0 = sys.argv[1]
    file1 = sys.argv[2]
    max_point = int(sys.argv[3])

    table0 = genfromtxt(file0, delimiter=',', skip_header=1)
    table1 = genfromtxt(file1, delimiter=',', skip_header=1)

    xmin = min(table0[:,0].min(), table1[:,0].min())
    xmax = min(table0[:,0].max(), table1[:,0].max())
    ymin = min(table0[:,1].min(), table1[:,1].min())
    ymax = min(table0[:,1].max(), table1[:,1].max())


    chunks = recursive_split(table0, table1, xmin, xmax, ymin, ymax, max_point)

    fname = "Chunks{}-{}_{}.txt"
    dataname = file0.split("0.")[0]

    for i, el in enumerate(chunks):
        save(fname, el[0], dataname, i+1, 0)
        save(fname, el[1], dataname, i+1, 1)

if __name__ == "__main__":
    main()
