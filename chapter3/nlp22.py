# 22. カテゴリ名の抽出
# 記事のカテゴリ名を（行単位ではなく名前で）抽出せよ
import json
import re
first = 3 + len('Category')
last = 2
text = ""
with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]

text = text.split("\n")
categories = [line for line in text if ('Category' in line)]

for category in categories:
    print(re.findall(r'^\[\[Category:(.*?)\]\]$', category))
    print(category[first:-last])
