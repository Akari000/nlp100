# 21. カテゴリ名を含む行を抽出
# 記事中でカテゴリ名を宣言している行を抽出せよ
import json
text = ""
with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]

text = text.split("\n")
categories = [line for line in text if ('Category' in line)]

for category in categories:
    print(category)
