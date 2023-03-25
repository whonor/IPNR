# -*- encoding:utf-8 -*-
"""
Date: create at 2020/10/2

Some help functions for building dataset
"""
import csv
import os
import pickle
import re
import json

import pandas as pd
import networkx as nx
import torch
import numpy as np

from random import sample

PADDING_NEWS = "<pad>"


def word_tokenize(sent):
    pat = re.compile(r"[\w]+|[|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def build_subgraph(graph: nx.DiGraph, vocab: list):
    all_ents = set([])
    all_ents = all_ents.union(vocab)
    all_ents = list(all_ents)
    sub_graph = graph.subgraph(all_ents)
    print(sub_graph)
    remove = [node for node, degree in dict(sub_graph.degree()).items() if degree < 1]
    print("Remove_nodes: ")
    print(remove)
    g = sub_graph.copy()
    g.remove_nodes_from(remove)
    subsub_graph = g.subgraph(sample(g.nodes(), 1000))  # 随机抽取50*20个节点，建立子图
    print("sample subgraph: ")
    print(subsub_graph)
    adj = nx.to_scipy_sparse_array(g).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    return edge_index, g.nodes, g


def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return ' '.join([k["WikidataId"] for k in json.loads(x)])


if __name__ == "__main__":
    mode = "dev"
    news_path = "../../MIND-small/"+mode+"/news.tsv"
    print("Loading concept info")
    news = pd.read_table(news_path,
                         sep='\t',
                         header=None,
                         usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                         quoting=csv.QUOTE_NONE,
                         names=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract', 'title_entities', 'abstract_entities', 'content'
                         ])
    df = news['content']
    df = df.fillna(" ")

    vocab = []

    for line in df:
        for w in word_tokenize(line):
            vocab.append(w)

    graph = pickle.load(open("../../data/MIND-small/concept_graph.pkl", 'rb'))

    edge_index, nodes, sub_graph = build_subgraph(graph=graph, vocab=vocab)  # vocab: 209090

    print(edge_index)
    print(sub_graph)
    pickle.dump(sub_graph, open("../../MIND-small/"+mode+"./concepts_subgraph.pkl", 'wb'))

    torch.save(edge_index, "../../data/MIND-small/"+mode+"/edge_index.pt")
    with open("../../MIND-small/"+mode+"/words_in_conceptnet.csv", 'w', encoding='utf-8') as f:
        for n in nodes:
            f.write(n)
            f.write('\n')
