
## build graph
### build conceptnet
准备ConceptNet数据放在conceptnet目录下，创建三元组数据。
- extract english words from Conceptnet 

`python3 01_extract_english_cpnet.py`

### build graph
使用networkx创建双向图，并生成子图和节点向量。

- build a concept graph based on Conceptnet
- generate edge_index for PYG

`python3 01_build_vocab_and_graph.py`

`python3 02_build_edge_index.py`

`python3 03_generate_concept_embeddings.py`


### TF-IDF
- use TF-IDF to build a new attribute "content" based on MIND dataset

创建新模式数据集，在此基础上执行TF_IDF

`python3 00_TFIDF_main.py`

## All Step
1) run 00_TF-IDF_main.py (build conceptnet);
2) run build_edge_index; build concept_embedding;
3) run data_preprocess;
4) run train.py.

