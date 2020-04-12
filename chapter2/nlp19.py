import pandas as pd

filename = "../data/popular-names.txt"
names = pd.read_csv(
    filename,
    names=("col1", "col2", "col3", "col4"),
    sep="\t",
    lineterminator="\n")

names = names.groupby("col1")
names = names.col1.count()
names = names.sort_values(ascending=False)

for row in names.index:
    print(row, end="\n")
