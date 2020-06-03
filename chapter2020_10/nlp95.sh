'''95. サブワード化Permalink
トークンの単位を単語や形態素からサブワードに変更し，91-94の実験を再度実施せよ．
'''
# test
subword-nmt learn-bpe -s 3000 < orig/kyoto-test.tokens.ja > codes.txt
subword-nmt apply-bpe -c codes.txt < orig/kyoto-test.tokens.ja > orig/kyoto-test.subword.ja

subword-nmt learn-bpe -s 3000 < orig/kyoto-test.tokens.en > codes.txt
subword-nmt apply-bpe -c codes.txt < orig/kyoto-test.tokens.en > orig/kyoto-test.subword.en

# valid
subword-nmt learn-bpe -s 3000 < orig/kyoto-dev.tokens.ja > codes.txt
subword-nmt apply-bpe -c codes.txt < orig/kyoto-dev.tokens.ja > orig/kyoto-dev.subword.ja

subword-nmt learn-bpe -s 3000 < orig/kyoto-dev.tokens.en > codes.txt
subword-nmt apply-bpe -c codes.txt < orig/kyoto-dev.tokens.en > orig/kyoto-dev.subword.en

# train
subword-nmt learn-bpe -s 3000 < orig/kyoto-train.tokens.ja > codes.txt
subword-nmt apply-bpe -c codes.txt < orig/kyoto-train.tokens.ja > orig/kyoto-train.subword.ja

subword-nmt learn-bpe -s 3000 < orig/kyoto-train.tokens.en > codes.txt
subword-nmt apply-bpe -c codes.txt < orig/kyoto-train.tokens.en > orig/kyoto-train.subword.en

# preprocess
onmt_preprocess -train_src orig/kyoto-train.subword.ja -train_tgt orig/kyoto-train.subword.en -valid_src orig/kyoto-dev.subword.ja -valid_tgt orig/kyoto-dev.subword.en -save_data data/subword
# train
onmt_train -data data/subword -save_model subword-model
# evaluate
onmt_translate -model subword-model_step_800.pt -src orig/kyoto-test.subword.ja -output subword-pred.txt -replace_unk -verbose
perl tools/multi-bleu.perl orig/kyoto-test.subword.en < pred.text