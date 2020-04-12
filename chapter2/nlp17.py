with open("../data/popular-names.txt") as f:
    lines = [line.split('\t')[0] for line in f.readlines()]

print(sorted(set(lines)))
