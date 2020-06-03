
# preprocess
onmt_preprocess -train_src dial/train.inputs -train_tgt dial/train.targets -valid_src dial/dev.inputs -valid_tgt dial/dev.targets -save_data data/dial
# train
onmt_train -data data/dial -save_model dial-model
# evaluate
onmt_translate -model dial-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src dial/test.inputs -output dial-pred.txt -replace_unk -verbose


'''
[2020-06-03 12:02:18,790 INFO] encoder: 29009000
[2020-06-03 12:02:18,791 INFO] decoder: 55812004
[2020-06-03 12:02:18,791 INFO] * number of parameters: 84821004
[2020-06-03 12:02:18,794 INFO] Starting training on CPU, could be very slow
[2020-06-03 12:02:18,794 INFO] Start training loop and validate every 10000 steps...
[2020-06-03 12:02:18,795 INFO] Loading dataset from data/dial.train.0.pt
[2020-06-03 12:02:27,983 INFO] number of examples: 468559
[2020-06-03 12:07:46,165 INFO] Step 50/  300; acc:   3.03; ppl: 1819946.21; xent: 14.41; lr: 1.00000; 159/166 tok/s;    327 sec
[2020-06-03 12:11:18,542 INFO] Step 100/  300; acc:   3.67; ppl: 248596.17; xent: 12.42; lr: 1.00000; 212/200 tok/s;    540 sec
[2020-06-03 12:14:00,638 INFO] Step 150/  300; acc:   4.36; ppl: 13140.12; xent: 9.48; lr: 1.00000; 233/289 tok/s;    702 sec
[2020-06-03 12:16:26,044 INFO] Step 200/  300; acc:   4.88; ppl: 2829.05; xent: 7.95; lr: 1.00000; 322/278 tok/s;    847 sec
[2020-06-03 12:19:21,285 INFO] Step 250/  300; acc:   4.54; ppl: 2176.82; xent: 7.69; lr: 1.00000; 286/296 tok/s;   1022 sec
[2020-06-03 12:21:41,109 INFO] Step 300/  300; acc:   6.20; ppl: 1734.67; xent: 7.46; lr: 1.00000; 318/277 tok/s;   1162 sec
'''