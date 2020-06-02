
# preprocess
onmt_preprocess -train_src dial/train.inputs -train_tgt dial/train.targets -valid_src dial/dev.inputs -valid_tgt dial/dev.targets -save_data data/dial
# train
onmt_train -data data/dial -save_model dial-model
# evaluate
onmt_translate -model dial-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src dial/test.inputs -output dial-pred.txt -replace_unk -verbose