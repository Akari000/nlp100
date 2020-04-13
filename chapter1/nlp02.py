# -*- coding: utf-8 -*-
text1 = u"パトカー"
text2 = u"タクシー"

for t1, t2 in zip(text1, text2):
    print(t1 + t2, end="")
