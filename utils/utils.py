def ngram(word_list, n, sep=""):
    ngram = []
    for i in range(0, len(word_list)-n+1):
        word = (sep).join(word_list[i:i+n])
        ngram.append(word)
    return ngram


def Union(X, Y):
    return set(X + Y)


def Difference(X, Y):
    return set(X) - set(Y)


def Intersection(X, Y):
    return set(X) & set(Y)