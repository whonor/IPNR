# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 下午 03:43
# @Author  : HonorWang
# @Email   : honorw@foxmail.com
# @File    : 00_TFIDF_main.py
# @Software: PyCharm
import csv
import re

import pandas as pd
import string
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

stops = set(stopwords.words("english"))
interpunctuations = str.maketrans('', '', string.punctuation)
table = str.maketrans('', '', string.digits)

ps = PorterStemmer()
wnl = WordNetLemmatizer()


def filter_stopwords(line: str):
    line = line.translate(table)
    line = word_tokenize(line.translate(interpunctuations).lower())
    line = ''.join(wnl.lemmatize(word) + '\t' for word in line if word not in stops)
    return line

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def word_tokenize(sent: str):
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def tfidf(words_list):
    vectorizer = CountVectorizer()
    if len(words_list) > 0:
        words_list = word_tokenize(words_list)

        word_frequence = vectorizer.fit_transform(words_list)
        words = vectorizer.get_feature_names()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(word_frequence)

    return tfidf, words


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=20):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def run_tfidf(words_list):
    tfidf_, words = tfidf(words_list)
    sorted_items = sort_coo(tfidf_.tocoo())
    results = extract_topn_from_vector(words, sorted_items, topn=20)

    return " ".join(list(results.keys()))


if __name__ == '__main__':
    source_train = "../../data/MIND-small/MINDsmall_train/news.tsv"
    train_news = pd.read_table(source_train,
                               header=None,
                               usecols=[0, 1, 2, 3, 4, 6, 7],
                               quoting=csv.QUOTE_NONE,
                               names=[
                                   'id', 'cate', 'subcate', 'title', 'abstract', 'title_entities', 'abstract_entities'
                               ])
    df = train_news
    df.fillna(" ")

    df['content'] = df['cate'] + '\t' + df['subcate'] + '\t' + df['title'] + '\t' + df['abstract']

    df['content'] = df['content'].apply(lambda x: filter_stopwords(str(x)))
    df['content'] = df['content'].apply(lambda x: run_tfidf(x))
    df.to_csv("../../data/MIND-small/train/news.tsv", header=False, index=False, sep='\t', quoting=3)

    print(df)
