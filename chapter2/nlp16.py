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
