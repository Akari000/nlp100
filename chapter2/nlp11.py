with open("../data/popular-names.txt") as f:
    for line in f.readlines():
        print(line.replace('\t', ' '), end="")
