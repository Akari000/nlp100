# 22. カテゴリ名の抽出
# 記事のカテゴリ名を（行単位ではなく名前で）抽出せよ
import json
import re
text = ""
with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]

categories = re.findall(r'\[\[Category:(.*?)(?:\|.*|)\]\]', text)

for category in categories:
    print(category)

'''
イギリス
英連邦王国
G8加盟国
欧州連合加盟国
海洋国家
君主国
島国
1801年に設立された州・地域
'''

'''note
() キャプチャ...中身を抽出
(?:) キャプチャしない丸括弧
'''
