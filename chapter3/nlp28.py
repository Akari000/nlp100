# 28. MediaWikiマークアップの除去
# 27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，国の基本情報を整形せよ．

import json
import re
from pprint import pprint
info = {}
with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]


text = re.findall(r'{{(基礎情報[\s\S]*\n)}}', text)[0]
text = re.sub(r"'{2,5}", '', text)  # 強調マークアップを除去
fields = re.findall(r'\|(.*?) = ([\s\S]*?)(?=\n\|)', text)

for field in fields:
    value = re.sub(
        r"\[\[(?:.*?\||)(.*?)\]\]", r'\1', field[1])  # 内部リンクマークアップを除去
    value = re.sub(r'<.*?>', '', value)  # htmlタグ，コメントアウトの除去
    value = re.sub(r'\[.*?\]', '', value)  # 外部リンクの除去
    value = re.sub(r'{{lang\|.*?\|(.*?)}}', r'\1', value)  # langの除去 {{lang|.*|value}} -> value
    value = re.sub(r'\*', '', value)    # 箇条書きの除去
    value = re.sub(r'.*?\|(.*?)', r'\1', value)    # フォントサイズの除去
    info[field[0]] = value

pprint(info)


"""
{'GDP/人': '36,727',
 'GDP値': '2兆3162億',
 'GDP値MER': '2兆4337億',
 'GDP値元': '1兆5478億',
 'GDP統計年': '2012',
 'GDP統計年MER': '2012',
 'GDP統計年元': '2012',
 'GDP順位': '6',
 'GDP順位MER': '5',
 'ISO 3166-1': 'GB / GBR',
 'ccTLD': '.uk / .gb使用は.ukに比べ圧倒的少数。',
 '人口値': '63,181,775',
 '人口大きさ': '1 E7',
 '人口密度値': '246',
 '人口統計年': '2011',
 '人口順位': '22',
 '位置画像': 'Location_UK_EU_Europe_001.svg',
 '元首等氏名': 'エリザベス2世',
 '元首等肩書': '女王',
 '公式国名': 'United Kingdom of Great Britain and Northern Ireland英語以外での正式国名:\n'
         'An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu '
         'Thuath（スコットランド・ゲール語）\n'
         'Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon（ウェールズ語）\n'
         'Ríocht Aontaithe na Breataine Móire agus Tuaisceart na '
         'hÉireann（アイルランド語）\n'
         'An Rywvaneth Unys a Vreten Veur hag Iwerdhon Glédh（コーンウォール語）\n'
         'Unitit Kinrick o Great Breetain an Northren Ireland（スコットランド語）\n'
         'Claught Kängrick o Docht Brätain an Norlin Airlann、Unitet Kängdom o '
         'Great Brittain an Norlin Airlann（アルスター・スコットランド語）',
 '公用語': '英語（事実上）',
 '国旗画像': 'Flag of the United Kingdom.svg',
 '国歌': '神よ女王陛下を守り給え',
 '国章リンク': '（国章）',
 '国章画像': 'イギリスの国章',
 '国際電話番号': '44',
 '夏時間': '+1',
 '建国形態': '建国',
 '日本語国名': 'グレートブリテン及び北アイルランド連合王国',
 '時間帯': '±0',
 '最大都市': 'ロンドン',
 '標語': 'Dieu et mon droit（フランス語:神と私の権利）',
 '水面積率': '1.3%',
 '略名': 'イギリス',
 '確立年月日1': '927年／843年',
 '確立年月日2': '1707年',
 '確立年月日3': '1801年',
 '確立年月日4': '1927年',
 '確立形態1': '1707年連合法まで）',
 '確立形態2': '1707年連合法）',
 '確立形態3': '1800年連合法）',
 '確立形態4': '現在の国号「グレートブリテン及び北アイルランド連合王国」に変更',
 '通貨': 'UKポンド (&pound;)',
 '通貨コード': 'GBP',
 '面積値': '244,820',
 '面積大きさ': '1 E11',
 '面積順位': '76',
 '首相等氏名': 'デーヴィッド・キャメロン',
 '首相等肩書': '首相',
 '首都': 'ロンドン'}
 """
