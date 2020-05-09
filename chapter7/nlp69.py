'''69. Webアプリケーションの作成
ユーザから入力された検索条件に合致するアーティストの情報を表示するWebアプリケーションを作成せよ．
アーティスト名，アーティストの別名，タグ等で検索条件を指定し，
アーティスト情報のリストをレーティングの高い順などで整列して表示せよ．
'''

from pymongo import MongoClient
from flask import Flask, render_template, request
app = Flask(__name__)

client = MongoClient('localhost', 27017)
db = client['nlp100_database']
collection = db['nlp100_collection']

query = {}


@app.route('/')
def home(query=query):
    documents = collection.find(filter=query).limit(20)
    return render_template(
        'index.html',
        documents=documents)


@app.route("/filter_by_name", methods=["POST"])
def filter_by_name():
    query['name'] = request.form["name"]
    return home(query)


@app.route("/filter_by_aliase", methods=["POST"])
def filter_by_aliase():
    query['aliases.name'] = request.form["aliase"]
    return home(query)


@app.route("/filter_by_tag", methods=["POST"])
def filter_by_tag():
    query['tags.value'] = request.form["tag"]
    return home(query)


@app.route("/clear", methods=["POST"])
def clear():
    query = {}
    return home(query)


if __name__ == "__main__":
    app.run(debug=True, port=8888, threaded=True)
