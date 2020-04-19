# 32. 動詞の原形
# 動詞の原形をすべて抽出せよ

import pickle

morpheme_list = []

with open('../data/neko.txt.mecab', 'rb') as f:
    morpheme_list = pickle.load(f)

for line in morpheme_list:
    if line['pos'] == '動詞':
        print(line['base'])


'''
れる
離れる
分る
れる
なる
いる
思う
てる
違う
いる
いる
なる
合う
なる
する
云う
見える
する
死ぬ
化ける
行く
する
穿く
鍛え上げる
乗り込む
くる
思う
なる
なる
なる
なる
する
する
する
合う
なる
合う
する
つく
する
いる
迎える
迎える
増す
くる
ある
落ちつく
する
保つ
いる
くる
かける
下がる
する
云う
分つ
くる
わかれる
云う
わく
れる
わかれる
れる
いる
する
いる
する
れる
くる
する
れる
のろける
云う
生れる
作る
ある
出る
いる
いる
騒ぐ
視る
話す
降る
する
滅する
滅する
陥る
しめる
払う
構う
する
せしめる
なる
縛する
られる
する
反す
する
知る
陥る
顧みる
達する
る
もつ
する
れる
得る
ある
ある
駆る
れる
挙げる
する
思い切る
叩く
云う
思う
する
する
する
する
する
錬る
する
いる
生れる
忘れる
出来る
あらわれる
云う
なる
分る
する
滅する
思う
云う
する
しまう
あきらめる
する
云う
云う
云う
出来る
する
ある
読む
云う
なる
作る
始まる
生れる
挙る
する
ある
する
出る
する
なる
読む
もつ
てる
作る
あらわれる
いる
あらわれる
見る
給う
見る
給う
ある
読む
する
なる
く
かく
わかる
なる
かく
わかる
なる
思う
れる
思う
れる
思う
知れる
出す
許す
許す
なる
担ぎ出す
やる
なくなる
する
見る
見える
する
すくむ
隣る
打てる
なる
かく
散らす
読む
云う
なる
ある
あつまる
出る
くる
見る
あらわす
写す
違う
かく
ある
ある
かえる
行く
出る
出る
立てる
利かす
いる
よる
知れる
威張る
利く
利く
振る
廻す
する
得る
得る
感じる
困る
いる
いう
反す
する
する
見る
給う
起す
つく
なる
云う
始める
する
する
化す
云う
出来る
悟る
悟る
もうす
罹る
飲む
考える
伺う
感じる
云う
云う
持つ
する
云う
出す
持つ
思う
飛ぶ
なる
読む
聞く
せる
聴く
持つ
来る
取り上げる
分る
てる
云う
驚く
いる
聞く
云う
いる
云う
ある
ある
這入る
聞く
聞く
なる
する
書く
ある
いう
かねる
聞く
てる
聞く
てる
聞く
てる
なる
とる
とる
なす
なす
なす
読む
問う
答える
書く
てる
振る
れる
出る
いる
問う
娶る
する
答える
ある
考える
ある
迂る
云う
云う
せる
恐る
入る
焼ける
入る
溺れる
行き詰る
逢う
とろける
出る
読む
御する
苦しめる
する
与る
うる
ある
起つ
あたう
至る
得る
もつ
する
する
似る
云う
飾る
もつ
蔽う
もとづく
おくる
告げる
忍ぶ
なす
得る
ある
垂れる
する
陥る
ある
くべる
かる
ある
ある
ある
似る
ある
棄てる
云う
する
ある
ある
聞く
する
からかう
掛ける
呼ぶ
する
.
.
.
'''
