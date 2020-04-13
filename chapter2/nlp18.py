import pandas as pd
filename = "../data/popular-names.txt"
names = pd.read_csv(
    filename,
    names=("col1", "col2", "col3", "col4"),
    sep='\t',
    lineterminator='\n')

names = names.sort_values('col3', ascending=False)
print(names.head(10).to_string(index=False, header=False))

#    Linda  F  99689  1947
#    Linda  F  96211  1948
#    James  M  94757  1947
#  Michael  M  92704  1957
#   Robert  M  91640  1947
#    Linda  F  91016  1949
#  Michael  M  90656  1956
#  Michael  M  90517  1958
#    James  M  88584  1948
#  Michael  M  88528  1954
