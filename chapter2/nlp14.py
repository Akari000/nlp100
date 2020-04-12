import sys
n = int(sys.argv[1])

with open("../data/popular-names.txt") as f:
    for i in range(n):
        print(f.readline().strip())
