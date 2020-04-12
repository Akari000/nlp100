import sys
n = int(sys.argv[1])
lines = []
with open("../data/popular-names.txt") as f:
    lines = f.readlines()

lines = lines[-n:]
for line in lines:
    print(line.strip())
