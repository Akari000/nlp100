'''94. ビーム探索
91で学習したニューラル機械翻訳モデルで翻訳文をデコードする際に，ビーム探索を導入せよ．
ビーム幅を1から100くらいまで適当に変化させながら，開発セット上のBLEUスコアの変化をプロットせよ．
'''
onmt_translate -model demo-model_step_1200.pt -src orig/kyoto-test.tokens.ja -output pred_beam1.txt -replace_unk -verbose --beam_size 1
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_beam1.text

onmt_translate -model demo-model_step_1200.pt -src orig/kyoto-test.tokens.ja -output pred_beam20.txt -replace_unk -verbose --beam_size 20
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_beam20.text

onmt_translate -model demo-model_step_1200.pt -src orig/kyoto-test.tokens.ja -output pred_beam40.txt -replace_unk -verbose --beam_size 40
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_beam40.text

'''
# beam 1
$ perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_beam1.txt
BLEU = 0.00, 2.0/0.1/0.0/0.0 (BP=1.000, ratio=1.116, hyp_len=30827, ref_len=27625)

$ perl tools/multi-bleu.perl orig/kyoto-test.tokens.ja < pred_beam1.txt
BLEU = 8.83, 24.6/11.1/6.5/3.4 (BP=1.000, ratio=1.168, hyp_len=30827, ref_len=26393)


# beam 5
$perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_beam5.txt
BLEU = 0.00, 1.9/0.1/0.0/0.0 (BP=0.988, ratio=0.988, hyp_len=27289, ref_len=27625)

$ perl tools/multi-bleu.perl orig/kyoto-test.tokens.ja < pred_beam5.txt
BLEU = 10.06, 27.5/12.6/7.5/3.9 (BP=1.000, ratio=1.034, hyp_len=27289, ref_len=26393)

'''