'''99. 翻訳サーバの構築
ユーザが翻訳したい文を入力すると，その翻訳結果がウェブブラウザ上で表示されるデモシステムを構築せよ
'''

import subprocess
from flask import Flask, render_template, request
import MeCab
import re
mt = MeCab.Tagger("-Owakati")
mt.parse('')
app = Flask(__name__)

MODEL_PATH = '/Users/hagaakari/develop/nlp100/data/kftt-data-1.0/data/demo-model_step_1200.pt'
USER_REPLY_PATH = '/Users/hagaakari/develop/nlp100/data/dial_user.txt'
SYSTEM_REPLY_PATH = '/Users/hagaakari/develop/nlp100/data/dial_system.txt'

dialogue = []


@app.route('/')
def home(dialogue=dialogue):
    return render_template(
        'index.html',
        dialogue=dialogue)


@app.route("/reply", methods=["POST"])
def reply():
    utter = {}
    utter['user'] = request.form["reply"]

    with open(USER_REPLY_PATH, 'w') as f:
        text = mt.parse(request.form["reply"])
        f.write(text)

    command = ['onmt_translate',
               '-model',
               MODEL_PATH,
               '-src',
               USER_REPLY_PATH,
               '-output',
               SYSTEM_REPLY_PATH,
               '-replace_unk',
               '-verbose']

    subprocess.call(command)
    with open(SYSTEM_REPLY_PATH, 'r') as f:
        text = f.read()

    utter['system'] = re.sub(' ', '', text)
    dialogue.append(utter)
    return home(dialogue)


if __name__ == "__main__":
    app.run(debug=True, port=8888, threaded=True)
