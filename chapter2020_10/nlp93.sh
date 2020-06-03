'''93. BLEUスコアの計測
91で学習したニューラル機械翻訳モデルの品質を調べるため，評価データにおけるBLEUスコアを測定せよ．
'''
onmt_translate -model demo-model_step_1200.pt -src orig/kyoto-test.tokens.ja -output pred.txt -replace_unk -verbose
# download multi-bleu.perl (https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl)
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred.text

'''
$ perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred.txt
BLEU = 0.00, 1.9/0.1/0.0/0.0 (BP=0.988, ratio=0.988, hyp_len=27289, ref_len=27625)

$ perl tools/multi-bleu.perl orig/kyoto-test.tokens.ja < pred.txt
BLEU = 10.06, 27.5/12.6/7.5/3.9 (BP=1.000, ratio=1.034, hyp_len=27289, ref_len=26393)
'''