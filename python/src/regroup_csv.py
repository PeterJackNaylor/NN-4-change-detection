
import pandas as pd
from glob import glob

files = glob("*.csv")
results = []
for f in files:
    results.append(pd.read_csv(f))

table = pd.concat(results, axis=0)
col = list(table.columns)
col[0] = "dataset"
table.columns = col
table.to_csv("benchmark.csv", index=False)