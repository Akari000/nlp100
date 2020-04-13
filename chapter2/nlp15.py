import sys
n = int(sys.argv[1])
lines = []
with open("../data/popular-names.txt") as f:
    lines = f.readlines()

lines = lines[-n:]
for line in lines:
    print(line.strip())

# Lucas   M       12585   2018
# Mason   M       12435   2018
# Logan   M       12352   2018
