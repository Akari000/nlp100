
# -*- coding: utf-8 -*-
str = u"パタトクカシーー"
newStr = ""

for i in range(0, len(u"パタトクカシーー"), 2):
    newStr += str[i+1]

print(newStr)
