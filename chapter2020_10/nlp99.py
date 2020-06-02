'''99. 翻訳サーバの構築
ユーザが翻訳したい文を入力すると，その翻訳結果がウェブブラウザ上で表示されるデモシステムを構築せよ
'''

from pymongo import MongoClient
import subprocess
from flask import Flask, render_template, request
app = Flask(__name__)

client = MongoClient('localhost', 27017)
db = client['nlp100_database']
collection = db['nlp100_collection']
MODEL_PATH = '../data/kftt-data-1.0/data/data/demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt'
OUT_PATH = './dial.txt'
user = []
system = []


@app.route('/')
def home(user=user, system=system):
    return render_template(
        'index.html',
        dialogue=zip(user, system))


@app.route("/reply", methods=["POST"])
def reply():
    user.append(request.form["reply"])

    with open(OUT_PATH, 'w') as f:
        f.write(request.form["reply"])

    command = ['onmt_translate',
               '-model',
               MODEL_PATH,
               '-src',
               OUT_PATH,
               '-replace_unk',
               '-verbose']
    text = subprocess.call(command)
    system.append(text)
    return home(user, system)


if __name__ == "__main__":
    app.run(debug=True, port=8888, threaded=True)
