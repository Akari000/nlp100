import re
import torch


def tokenize(doc):
    tokens = doc.split(' ')
    return tokens


def normalize(doc):
    doc = re.sub(r"[',.]", '', doc)   # 記号を削除
    doc = re.sub(r" {2,}", ' ', doc)  # 2回以上続くスペースを削除
    doc = re.sub(r" *?$", '', doc)    # 行頭と行末のスペースを削除
    doc = re.sub(r"^ *?", '', doc)
    doc = doc.lower()                 # 小文字に統一
    return doc


def preprocessor(doc):
    doc = normalize(doc)
    tokens = tokenize(doc)
    return tokens
