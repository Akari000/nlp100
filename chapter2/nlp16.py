import sys
n = int(sys.argv[1])
lines = []
with open("../data/popular-names.txt") as f:
    lines = f.readlines()
size = len(lines) / n
size = int(size)

for i in range(size):
    print(('').join(lines[(i*n):i*n+n]))

print(('').join(lines[(n*size):]))

# Mary    F       7065    1880
# Anna    F       2604    1880
# Emma    F       2003    1880

# Elizabeth       F       1939    1880
# Minnie  F       1746    1880
# Margaret        F       1578    1880

# Ida     F       1472    1880
# Alice   F       1414    1880
# Bertha  F       1320    1880

# Sarah   F       1288    1880
# John    M       9655    1880
# William M       9532    1880

# James   M       5927    1880
# Charles M       5348    1880
# George  M       5126    1880

# Frank   M       3242    1880
# Joseph  M       2632    1880
# Thomas  M       2534    1880

# Henry   M       2444    1880
# Robert  M       2415    1880
# Mary    F       6919    1881

# Anna    F       2698    1881
# Emma    F       2034    1881
# Elizabeth       F       1852    1881

# Margaret        F       1658    1881
# Minnie  F       1653    1881
# Ida     F       1439    1881

# Annie   F       1326    1881
# Bertha  F       1324    1881
# Alice   F       1308    1881
# ...