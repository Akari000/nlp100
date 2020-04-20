# 21. カテゴリ名を含む行を抽出
# 記事中でカテゴリ名を宣言している行を抽出せよ
import json
import re


text = ""
with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]


categories = re.findall(r'\[\[Category:.*?\]\]', text)
for category in categories:
    print(category)


'''
[[Category:イギリス|*]]
[[Category:英連邦王国|*]]
[[Category:G8加盟国]]
[[Category:欧州連合加盟国]]
[[Category:海洋国家]]
[[Category:君主国]]
[[Category:島国|くれいとふりてん]]
[[Category:1801年に設立された州・地域]]
'''

'''note
.* 全ての文字
.+ 空文字を含まない全ての文字
? 非貪欲
'''
