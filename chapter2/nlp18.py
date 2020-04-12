import pandas as pd
filename = "../data/popular-names.txt"
names = pd.read_csv(
    filename,
    names=("col1", "col2", "col3", "col4"),
    sep='\t',
    lineterminator='\n')

names = names.sort_values('col3', ascending=False)
print(names.head(10).to_string(index=False, header=False))
