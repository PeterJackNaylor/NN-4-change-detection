import pandas as pd
import sys


table = pd.read_csv(sys.argv[1])
table["datacloud"] = table.name.apply(lambda row: row.split("__")[0])

def datasegment_f(x):
    datasegment = x.split("-")[0]
    if "tmp" == datasegment[:3]:
        datasegment = datasegment[:-1]
    return datasegment

table["datasegment"] = table.datacloud.apply(lambda row: datasegment_f(row))

table["method"] = table.name.apply(lambda row: row.split("_")[-1])
idx = table.groupby(["datacloud", "method"])["score"].transform(min) == table["score"]

table = table.loc[idx]
table[["name", "datasegment", "method"]].to_csv("selected.csv", index=False)
