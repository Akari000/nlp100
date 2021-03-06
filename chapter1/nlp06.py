import sys
sys.path.append('../utils')
from utils import ngram
textX = "paraparaparadise"
textY = "paragraph"


def Union(X, Y):
    # set(X) | set(Y) を使っても良い
    return set(X + Y)


def Difference(X, Y):
    return set(X) - set(Y)


def Intersection(X, Y):
    return set(X) & set(Y)


X = ngram(textX, 2)
Y = ngram(textY, 2)
print("X", X)
print("Y", Y)

print("和集合", Union(X, Y))
print("差集合", Difference(X, Y))
print("積集合", Intersection(X, Y))

print("se in X: ", "se" in X)
print("se in Y: ", "se" in Y)
