# 24. ファイル参照の抽出
# 記事から参照されているメディアファイルをすべて抜き出せ．


import json
import re
text = ""


with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]

files = re.findall(r'\[\[(?:ファイル|File):(.*?)(?:\|.*?)\]\]', text)
# ファイルの参照例  [[ファイル:Wikipedia-logo-v2-ja.png|thumb|説明文]]
# File: または ファイル: から始まるもの
# |以降は除去

for f in files:
    print(f)

'''
Royal Coat of Arms of the United Kingdom.svg
Battle of Waterloo 1815.PNG
The British Empire.png
Uk topo en.jpg
BenNevis2005.jpg
Elizabeth II greets NASA GSFC employees, May 8, 2007 edit.jpg
Palace of Westminster, London - Feb 2007.jpg
David Cameron and Barack Obama at the G20 Summit in Toronto.jpg
Soldiers Trooping the Colour, 16th June 2007.jpg
Scotland Parliament Holyrood.jpg
London.bankofengland.arp.jpg
City of London skyline from London City Hall - Oct 2008.jpg
Oil platform in the North SeaPros.jpg
Eurostar at St Pancras Jan 2008.jpg
Heathrow T5.jpg
Anglospeak.svg
CHANDOS3.jpg
The Fabs.JPG
Wembley Stadium, illuminated.jpg
'''

'''note
一文字一致の時は[]
文字列一致の時は()
'''
