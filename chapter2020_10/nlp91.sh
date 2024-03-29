'''
91. 機械翻訳モデルの訓練
90で準備したデータを用いて，ニューラル機械翻訳のモデルを学習せよ
（ニューラルネットワークのモデルはTransformerやLSTMなど適当に選んでよい）．
'''

onmt_preprocess -train_src orig/kyoto-train.tokens.ja -train_tgt orig/kyoto-train.tokens.en -valid_src orig/kyoto-dev.tokens.ja -valid_tgt orig/kyoto-dev.tokens.en -save_data data/demo
onmt_train -data data/demo -save_model demo-model --log_file demo-model-log

'''生成されるファイル
data/demo.train.0.pt
data/demo.test.0.pt
data/demo.dev.0.pt
'''

'''demo-model-log
[2020-06-02 23:59:12,589 INFO]  * src vocab size = 50002
[2020-06-02 23:59:12,590 INFO]  * tgt vocab size = 50004
[2020-06-02 23:59:12,590 INFO] Building model...
[2020-06-02 23:59:13,465 INFO] NMTModel(
  (encoder): RNNEncoder(
    (embeddings): Embeddings(
      (make_embedding): Sequential(
        (emb_luts): Elementwise(
          (0): Embedding(50002, 500, padding_idx=1)
        )
      )
    )
    (rnn): LSTM(500, 500, num_layers=2, dropout=0.3)
  )
  (decoder): InputFeedRNNDecoder(
    (embeddings): Embeddings(
      (make_embedding): Sequential(
        (emb_luts): Elementwise(
          (0): Embedding(50004, 500, padding_idx=1)
        )
      )
    )
    (dropout): Dropout(p=0.3, inplace=False)
    (rnn): StackedLSTM(
      (dropout): Dropout(p=0.3, inplace=False)
      (layers): ModuleList(
        (0): LSTMCell(1000, 500)
        (1): LSTMCell(500, 500)
      )
    )
    (attn): GlobalAttention(
      (linear_in): Linear(in_features=500, out_features=500, bias=False)
      (linear_out): Linear(in_features=1000, out_features=500, bias=False)
    )
  )
  (generator): Sequential(
    (0): Linear(in_features=500, out_features=50004, bias=True)
    (1): Cast()
    (2): LogSoftmax()
  )
)
[2020-06-02 23:59:13,465 INFO] encoder: 29009000
[2020-06-02 23:59:13,465 INFO] decoder: 55812004
[2020-06-02 23:59:13,465 INFO] * number of parameters: 84821004
[2020-06-02 23:59:13,466 INFO] Starting training on CPU, could be very slow
[2020-06-02 23:59:13,466 INFO] Start training loop and validate every 10000 steps...
[2020-06-02 23:59:13,467 INFO] Loading dataset from data/demo.train.0.pt
[2020-06-02 23:59:14,975 INFO] number of examples: 90173
[2020-06-03 00:02:59,784 INFO] Step 50/ 5000; acc:   2.94; ppl: 5541291.14; xent: 15.53; lr: 1.00000; 304/318 tok/s;    226 sec
[2020-06-03 00:07:08,742 INFO] Step 100/ 5000; acc:   4.12; ppl: 23939.60; xent: 10.08; lr: 1.00000; 288/301 tok/s;    475 sec
[2020-06-03 00:11:19,018 INFO] Step 150/ 5000; acc:   4.35; ppl: 5674.64; xent: 8.64; lr: 1.00000; 303/316 tok/s;    726 sec
[2020-06-03 00:14:51,231 INFO] Step 200/ 5000; acc:   5.97; ppl: 4145.55; xent: 8.33; lr: 1.00000; 301/316 tok/s;    938 sec
[2020-06-03 00:18:42,311 INFO] Step 250/ 5000; acc:   8.43; ppl: 2167.23; xent: 7.68; lr: 1.00000; 288/302 tok/s;   1169 sec
[2020-06-03 00:22:54,488 INFO] Step 300/ 5000; acc:   8.06; ppl: 1701.45; xent: 7.44; lr: 1.00000; 291/303 tok/s;   1421 sec
[2020-06-03 00:26:12,175 INFO] Step 350/ 5000; acc:   9.64; ppl: 1368.08; xent: 7.22; lr: 1.00000; 328/345 tok/s;   1619 sec
[2020-06-03 00:29:34,777 INFO] Step 400/ 5000; acc:  12.40; ppl: 988.48; xent: 6.90; lr: 1.00000; 332/348 tok/s;   1821 sec
[2020-06-03 00:33:23,534 INFO] Step 450/ 5000; acc:  11.86; ppl: 914.36; xent: 6.82; lr: 1.00000; 329/343 tok/s;   2050 sec
[2020-06-03 00:37:02,174 INFO] Step 500/ 5000; acc:  14.25; ppl: 766.55; xent: 6.64; lr: 1.00000; 319/334 tok/s;   2269 sec
[2020-06-03 00:40:47,631 INFO] Step 550/ 5000; acc:  14.97; ppl: 702.34; xent: 6.55; lr: 1.00000; 308/322 tok/s;   2494 sec
[2020-06-03 00:44:36,345 INFO] Step 600/ 5000; acc:  16.35; ppl: 598.30; xent: 6.39; lr: 1.00000; 294/308 tok/s;   2723 sec
[2020-06-03 00:48:25,901 INFO] Step 650/ 5000; acc:  16.85; ppl: 580.51; xent: 6.36; lr: 1.00000; 294/308 tok/s;   2952 sec
[2020-06-03 00:52:02,225 INFO] Step 700/ 5000; acc:  18.09; ppl: 508.28; xent: 6.23; lr: 1.00000; 301/316 tok/s;   3169 sec
[2020-06-03 00:55:37,438 INFO] Step 750/ 5000; acc:  19.06; ppl: 442.98; xent: 6.09; lr: 1.00000; 294/309 tok/s;   3384 sec
[2020-06-03 00:59:19,111 INFO] Step 800/ 5000; acc:  19.66; ppl: 433.82; xent: 6.07; lr: 1.00000; 304/318 tok/s;   3606 sec
[2020-06-03 01:03:02,263 INFO] Step 850/ 5000; acc:  21.05; ppl: 376.78; xent: 5.93; lr: 1.00000; 295/309 tok/s;   3829 sec
[2020-06-03 01:07:25,603 INFO] Step 900/ 5000; acc:  20.53; ppl: 383.26; xent: 5.95; lr: 1.00000; 297/309 tok/s;   4092 sec
[2020-06-03 01:10:53,225 INFO] Step 950/ 5000; acc:  23.30; ppl: 318.57; xent: 5.76; lr: 1.00000; 293/308 tok/s;   4300 sec
[2020-06-03 01:14:45,751 INFO] Step 1000/ 5000; acc:  22.94; ppl: 311.04; xent: 5.74; lr: 1.00000; 298/312 tok/s;   4532 sec
[2020-06-03 01:18:35,723 INFO] Step 1050/ 5000; acc:  23.45; ppl: 302.20; xent: 5.71; lr: 1.00000; 297/311 tok/s;   4762 sec
[2020-06-03 01:22:32,253 INFO] Step 1100/ 5000; acc:  23.23; ppl: 293.54; xent: 5.68; lr: 1.00000; 296/310 tok/s;   4999 sec
[2020-06-03 01:27:06,424 INFO] Step 1150/ 5000; acc:  24.53; ppl: 271.13; xent: 5.60; lr: 1.00000; 266/278 tok/s;   5273 sec
[2020-06-03 01:30:35,797 INFO] Step 1200/ 5000; acc:  25.16; ppl: 261.22; xent: 5.57; lr: 1.00000; 324/340 tok/s;   5482 sec
[2020-06-03 01:34:29,435 INFO] Step 1250/ 5000; acc:  25.89; ppl: 239.69; xent: 5.48; lr: 1.00000; 321/335 tok/s;   5716 sec
[2020-06-03 01:38:25,877 INFO] Step 1300/ 5000; acc:  26.43; ppl: 236.73; xent: 5.47; lr: 1.00000; 301/314 tok/s;   5952 sec
[2020-06-03 01:41:56,670 INFO] Step 1350/ 5000; acc:  27.78; ppl: 219.32; xent: 5.39; lr: 1.00000; 324/339 tok/s;   6163 sec
[2020-06-03 01:45:19,818 INFO] Step 1400/ 5000; acc:  29.29; ppl: 199.70; xent: 5.30; lr: 1.00000; 310/326 tok/s;   6366 sec
[2020-06-03 01:46:07,330 INFO] Loading dataset from data/demo.train.0.pt
[2020-06-03 01:46:09,896 INFO] number of examples: 90173
[2020-06-03 01:49:16,432 INFO] Step 1450/ 5000; acc:  30.16; ppl: 182.59; xent: 5.21; lr: 1.00000; 291/305 tok/s;   6603 sec
[2020-06-03 01:53:21,439 INFO] Step 1500/ 5000; acc:  31.20; ppl: 175.21; xent: 5.17; lr: 1.00000; 297/310 tok/s;   6848 sec
[2020-06-03 01:57:38,956 INFO] Step 1550/ 5000; acc:  33.77; ppl: 149.21; xent: 5.01; lr: 1.00000; 287/299 tok/s;   7105 sec
[2020-06-03 02:01:14,732 INFO] Step 1600/ 5000; acc:  37.35; ppl: 119.18; xent: 4.78; lr: 1.00000; 303/318 tok/s;   7321 sec
[2020-06-03 02:05:07,542 INFO] Step 1650/ 5000; acc:  42.54; ppl: 87.12; xent: 4.47; lr: 1.00000; 295/309 tok/s;   7554 sec
[2020-06-03 02:08:56,628 INFO] Step 1700/ 5000; acc:  46.24; ppl: 68.53; xent: 4.23; lr: 1.00000; 297/311 tok/s;   7783 sec
[2020-06-03 02:12:26,451 INFO] Step 1750/ 5000; acc:  52.22; ppl: 42.79; xent: 3.76; lr: 1.00000; 336/351 tok/s;   7993 sec
[2020-06-03 02:15:46,729 INFO] Step 1800/ 5000; acc:  58.66; ppl: 25.43; xent: 3.24; lr: 1.00000; 336/352 tok/s;   8193 sec
[2020-06-03 02:19:25,271 INFO] Step 1850/ 5000; acc:  60.24; ppl: 21.78; xent: 3.08; lr: 1.00000; 338/352 tok/s;   8412 sec
[2020-06-03 03:16:24,115 INFO] Step 1900/ 5000; acc:  64.22; ppl: 15.48; xent: 2.74; lr: 1.00000;  19/ 20 tok/s;  11831 sec
[2020-06-03 03:20:52,013 INFO] Step 1950/ 5000; acc:  65.64; ppl: 13.71; xent: 2.62; lr: 1.00000; 271/283 tok/s;  12099 sec
[2020-06-03 03:32:06,459 INFO] Step 2000/ 5000; acc:  67.06; ppl: 12.17; xent: 2.50; lr: 1.00000;  99/104 tok/s;  12773 sec
[2020-06-03 05:45:56,634 INFO] Step 2050/ 5000; acc:  68.44; ppl: 10.93; xent: 2.39; lr: 1.00000;   9/  9 tok/s;  20803 sec
[2020-06-03 05:49:43,372 INFO] Step 2100/ 5000; acc:  69.49; ppl: 10.02; xent: 2.30; lr: 1.00000; 279/293 tok/s;  21030 sec
[2020-06-03 05:53:44,116 INFO] Step 2150/ 5000; acc:  70.11; ppl:  9.29; xent: 2.23; lr: 1.00000; 258/271 tok/s;  21271 sec
[2020-06-03 06:07:56,726 INFO] Step 2200/ 5000; acc:  70.26; ppl:  9.17; xent: 2.22; lr: 1.00000;  81/ 85 tok/s;  22123 sec
[2020-06-03 06:54:13,611 INFO] Step 2250/ 5000; acc:  72.25; ppl:  7.81; xent: 2.05; lr: 1.00000;  22/ 23 tok/s;  24900 sec
[2020-06-03 06:58:12,610 INFO] Step 2300/ 5000; acc:  72.87; ppl:  7.57; xent: 2.02; lr: 1.00000; 331/344 tok/s;  25139 sec
[2020-06-03 07:01:28,203 INFO] Step 2350/ 5000; acc:  72.65; ppl:  7.62; xent: 2.03; lr: 1.00000; 341/357 tok/s;  25335 sec


'''