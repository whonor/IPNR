# encoding: utf-8
"""
Build graph
"""
import csv
import os
import json
import pickle
import argparse

import pandas as pd
import numpy as np
import networkx as nx
from graph.vocab import WordVocab


def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return ' '.join([k["WikidataId"] for k in json.loads(x)])


def build_entire_graph(kg_path="../conceptnet/conceptnet.en.csv"):
    edges = list()
    wikigraph_file = open(kg_path, 'r', encoding='utf-8')
    for triple in wikigraph_file:
        triplesplit = triple.strip().split('\t')
        edges.append([triplesplit[1], triplesplit[2], float(triplesplit[3])])

    edge_df = pd.DataFrame(edges, columns=["from", "to", "weight"])
    edge_weights = edge_df.groupby(["from", "to"]).apply(lambda x: sum(x["weight"]))
    weighted_edges = edge_weights.to_frame().reset_index().values
    dg = nx.DiGraph()
    dg.add_weighted_edges_from(weighted_edges)
    print(dg)

    return dg


def main(cfg):
    print("Building Graph")
    graph = build_entire_graph()
    print("Original Graph built from all concepts:", len(graph.nodes))
    print(graph)
    graph_path = os.path.join(ROOT_PATH, "concept_graph.pkl")
    pickle.dump(graph, open(graph_path, 'wb'))


if __name__ == "__main__":
    ROOT_PATH = "../../data/MIND-small"
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default=80000, type=int,
                        help="Path of the validation data file.")
    parser.add_argument("--lower", action='store_true')
    args = parser.parse_args()
    main(args)
