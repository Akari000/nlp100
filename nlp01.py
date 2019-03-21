
# -*- coding: utf-8 -*-
str = u"パタトクカシーー"
newStr = ""

for i in range(0, len(u"パタトクカシーー"), 2):
    print(i)
    newStr += str[i+1]
    print(str[i].encode('utf-8'))

print(newStr.encode('utf-8'))
