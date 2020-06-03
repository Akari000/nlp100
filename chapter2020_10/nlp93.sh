'''93. BLEUスコアの計測
91で学習したニューラル機械翻訳モデルの品質を調べるため，評価データにおけるBLEUスコアを測定せよ．
'''
onmt_translate -model demo-model_step_800.pt -src orig/kyoto-test.tokens.en -output pred.txt -replace_unk -verbose
# download multi-bleu.perl (https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl)
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred.text

'''
$ perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred.txt
BLEU = 0.00, 4.7/0.1/0.0/0.0 (BP=0.953, ratio=0.954, hyp_len=26358, ref_len=27625)

$ perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_en.txt
BLEU = 0.80, 21.7/2.2/0.2/0.0 (BP=0.996, ratio=0.996, hyp_len=27528, ref_len=27625)

$ perl tools/multi-bleu.perl orig/kyoto-test.tokens.ja < pred_en.txt
BLEU = 0.00, 0.9/0.0/0.0/0.0 (BP=1.000, ratio=1.043, hyp_len=27528, ref_len=26393)
'''