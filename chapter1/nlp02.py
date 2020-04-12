# -*- coding: utf-8 -*-
str1 = u"パトカー"
str2 = u"タクシー"
result = ""

for i in range(0, len(str1)):
    result = result + str1[i] + str2[i]

print(result)
