with open("../data/popular-names.txt") as f:
    lines = f.readlines()
    col1 = [line.split('\t')[0] for line in lines]
    with open("../data/col1.txt", "a") as out:
        out.write(('\n').join(col1))

    col2 = [line.split('\t')[1] for line in lines]
    with open("../data/col2.txt", "a") as out:
        out.write(('\n').join(col2))
