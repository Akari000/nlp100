text1 = []
text2 = []

with open("../data/col1.txt") as f:
    text1 = f.readlines()
with open("../data/col2.txt") as f:
    text2 = f.readlines()

with open("../data/nlp13.txt", "w") as f:
    for t1, t2 in zip(text1, text2):
        f.write("%s\t%s\n" % (t1.strip(), t2.strip()))

# nlp13.text
# Mary	F
# Anna	F
# Emma	F
# Elizabeth	F
# Minnie	F
# Margaret	F
# Ida	F
# Alice	F
# ...
