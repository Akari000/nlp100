'''97. ハイパー・パラメータの調整
ニューラルネットワークのモデルや，そのハイパーパラメータを変更しつつ，
開発データにおけるBLEUスコアが最大となるモデルとハイパーパラメータを求めよ．
'''

# TODO pythonファイルからコマンドを打ってoptunaを使う

onmt_train -data data/demo -save_model demo-model -optim adam -learning_rate 0.1 --rnn_type GRU
onmt_translate -model demo-model_step_1200.pt -src orig/kyoto-test.tokens.ja -output pred.txt -replace_unk -verbose
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred.text

onmt_train -data data/demo -save_model demo-model -optim sgd -learning_rate 0.1 --rnn_type GRU
onmt_translate -model demo-model_step_1200.pt -src orig/kyoto-test.tokens.ja -output pred.txt -replace_unk -verbose
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred.text

onmt_train -data data/demo -save_model demo-model -optim adam -learning_rate 0.1 --rnn_type LSTM
onmt_translate -model demo-model_step_1200.pt -src orig/kyoto-test.tokens.ja -output pred.txt -replace_unk -verbose
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred.text

onmt_train -data data/demo -save_model demo-model -optim adam -learning_rate 0.01 --rnn_type GRU
onmt_translate -model demo-model_step_1200.pt -src orig/kyoto-test.tokens.ja -output pred.txt -replace_unk -verbose
perl tools/multi-bleu.perl orig/kyoto-test.tokens.en < pred.text

