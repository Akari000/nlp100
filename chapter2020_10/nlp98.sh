
# preprocess
onmt_preprocess -train_src train.inputs -train_tgt train.targets -valid_src dev.inputs -valid_tgt dev.targets -save_data data/demo
# train
onmt_train -data data/demo -save_model demo-model
# evaluate
onmt_translate -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src orig/kyoto-test.subword.ja -output pred.txt -replace_unk -verbose