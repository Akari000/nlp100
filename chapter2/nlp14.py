import sys
n = int(sys.argv[1])

with open("../data/popular-names.txt") as f:
    for i in range(n):
        print(f.readline().strip())

# (n=3)
# Mary    F       7065    1880
# Anna    F       2604    1880
# Emma    F       2003    1880
