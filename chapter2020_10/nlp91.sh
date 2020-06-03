'''
91. 機械翻訳モデルの訓練
90で準備したデータを用いて，ニューラル機械翻訳のモデルを学習せよ
（ニューラルネットワークのモデルはTransformerやLSTMなど適当に選んでよい）．
'''

onmt_preprocess -train_src orig/kyoto-train.tokens.ja -train_tgt orig/kyoto-train.tokens.en -valid_src orig/kyoto-dev.tokens.ja -valid_tgt orig/kyoto-dev.tokens.en -save_data data/demo
onmt_train -data data/demo -save_model demo-model

'''outputs
[2020-06-02 22:24:08,051 INFO] Extracting features...
[2020-06-02 22:24:08,052 INFO]  * number of source features: 0.
[2020-06-02 22:24:08,052 INFO]  * number of target features: 0.
[2020-06-02 22:24:08,052 INFO] Building `Fields` object...
[2020-06-02 22:24:08,053 INFO] Building & saving training data...
[2020-06-02 22:24:43,569 INFO]  * tgt vocab size: 50004.
[2020-06-02 22:24:43,765 INFO]  * src vocab size: 50002.
[2020-06-02 22:24:44,308 INFO] Building & saving validation data...

生成されるファイル
data/demo.train.0.pt
data/demo.test.0.pt
data/demo.dev.0.pt

'''