def word_ngram(wordList, n):
    ngram = []
    for i in range(0, len(wordList)-1):
        word = (" ").join(wordList[i:i+n])
        ngram.append(word)
    return ngram


def ch_ngram(wordList, n):
    ngram = []
    for i in range(0, len(wordList)-1):
        word = ("").join(wordList[i:i+n])
        ngram.append(word)
    return ngram


def Union(X, Y):
    return set(X + Y)


def Difference(X, Y):
    return set(X) - set(Y)


def Intersection(X, Y):
    return set(X) & set(Y)