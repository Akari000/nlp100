# -*- coding: utf-8 -*-
str1 = u"パトカー"
str2 = u"タクシー"

newStr = ""

for i in range(0, len(str1)):
    newStr = newStr + str1[i] + str2[i]

print(newStr)
print(newStr.encode('utf-8'))
