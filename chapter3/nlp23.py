# 23. セクション構造
# 記事中に含まれるセクション名とそのレベル（例えば"== セクション名 =="なら1）を表示せよ

import json
import re
text = ""
with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]

# TODO 正規表現を使う
# re.compileでpatternを指定できる．

sections = re.findall(r'(={2,5})(?: |)(.*?)={2,5}', text)

for section in sections:
    level = len(section[0]) - 1
    name = section[1]
    print(name, level)

# 国名 1
# 歴史 1
# 地理 1
# 気候 2
# 政治 1
# 外交と軍事 1
# 地方行政区分 1
# 主要都市 2
# 科学技術 1
# 経済 1
# 鉱業 2
# 農業 2
# 貿易 2
# 通貨 2
# 企業 2
# 交通 1
# 道路 2
# 鉄道 2
# 海運 2
# 航空 2
# 通信 1
# 国民 1
# 言語 2
# 宗教 2
# 婚姻 2
# 教育 2
# 文化 1
# 食文化 2
# 文学 2
# 哲学 2
# 音楽 2
# イギリスのポピュラー音楽 3
# 映画 2
# コメディ 2
# 国花 2
# 世界遺産 2
# 祝祭日 2
# スポーツ 1
# サッカー 2
# 競馬 2
# モータースポーツ 2
# 脚注 1
# 関連項目 1
# 外部リンク 1

'''note
{n,m} 直前の正規表現を m 回から n 回、できるだけ多く繰り返したものにマッチ
'''
