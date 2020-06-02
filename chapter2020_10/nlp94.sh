onmt_train -data data/demo -save_model demo-model_1 --beam_size 1
onmt_train -data data/demo -save_model demo-model_20 --beam_size 20
onmt_train -data data/demo -save_model demo-model_40 --beam_size 40

# onmt_translate -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src orig/kyoto-test.tokens.ja -output pred.txt -replace_unk -verbose
# perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred.text