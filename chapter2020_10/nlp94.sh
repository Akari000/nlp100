'''94. ビーム探索
91で学習したニューラル機械翻訳モデルで翻訳文をデコードする際に，ビーム探索を導入せよ．
ビーム幅を1から100くらいまで適当に変化させながら，開発セット上のBLEUスコアの変化をプロットせよ．
'''
onmt_translate -model demo-model_1_step_800.pt -src orig/kyoto-test.tokens.ja -output pred_beam1.txt -replace_unk -verbose --beam_size 1
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_beam1.text

onmt_translate -model demo-model_20_step_800.pt -src orig/kyoto-test.tokens.ja -output pred_beam20.txt -replace_unk -verbose --beam_size 20
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_beam20.text

onmt_translate -model demo-model_step_800.pt -src orig/kyoto-test.tokens.ja -output pred_beam40.txt -replace_unk -verbose --beam_size 40
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_beam40.text

'''
$ perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_beam1_en.txt
BLEU = 0.86, 22.4/2.2/0.2/0.0 (BP=1.000, ratio=1.002, hyp_len=27692, ref_len=27625)

$ perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_beam20_en.txt
BLEU = 0.80, 21.7/2.2/0.2/0.0 (BP=0.996, ratio=0.996, hyp_len=27528, ref_len=27625)

$ perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred_beam1_en.txt
BLEU = 0.80, 21.7/2.2/0.2/0.0 (BP=0.996, ratio=0.996, hyp_len=27528, ref_len=27625)

'''