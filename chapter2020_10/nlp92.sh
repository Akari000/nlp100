'''92. 機械翻訳モデルの適用
91で学習したニューラル機械翻訳モデルを用い，
与えられた（任意の）日本語の文を英語に翻訳するプログラムを実装せよ
'''
onmt_translate -model demo-model_step_1200.pt -src orig/kyoto-test.tokens.ja -output pred.txt -replace_unk -verbose

"""output

SENT 984: ['現', '門', '首', 'は', '大谷', '暢', '顕', '（', '浄', '如', '）', '。']
PRED 984: 現 ( 門 ) ( 門 ) 浄 ( 如 ) )
PRED SCORE: -6.4147

SENT 985: ['国宝']
PRED 985: 国宝
PRED SCORE: -0.0330

SENT 986: ['重要', '文化財']
PRED 986: 重要 文化財
PRED SCORE: -0.1829

SENT 987: ['京都', '市営', '地下鉄', '京都', '市営', '地下鉄', '五条', '駅', '(', '京都', '市営', '地下鉄', ')', '下車']
PRED 987: 京都 ( 市営 ) 下車 ( 下車 ) 下車 下車 ( 下車 ) )
PRED SCORE: -7.3149

SENT 988: ['大学']
PRED 988: 大学
PRED SCORE: -0.0330

SENT 989: ['大谷大学']
PRED 989: 大谷大学
PRED SCORE: -0.0330

日本語→英語にならない．最初の単語はそのまま返している．繰り返しが多い．
"""