
import torch
from torchtext.vocab import GloVe
from tqdm import tqdm
import pandas as pd
import numpy as np

word_embedding_dim = 300
#  Glove word embedding
train_concepts = pd.read_csv("../../MIND-small/train/words_in_conceptnet.csv", encoding="utf-8",
                             names=["word"])
test_concepts = pd.read_csv("../../MIND-small/test/words_in_conceptnet.csv", encoding="utf-8",
                            names=["word"])
all_concepts = pd.concat([train_concepts, test_concepts], join='inner', ignore_index=True).drop_duplicates()

word_dict = {'<PAD>': 0, '<UNK>': 1}

for i, word in enumerate(all_concepts['word']):
    word_dict[word] = i + 2

if word_embedding_dim == 300:
    glove = GloVe(name='840B', dim=300, cache='../../glove', max_vectors=10000000000)
else:
    glove = GloVe(name='6B', dim=word_embedding_dim, cache='../glove', max_vectors=10000000000)
glove_stoi = glove.stoi
glove_vectors = glove.vectors
glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
word_embedding_vectors = torch.zeros([len(word_dict), word_embedding_dim])
for word in tqdm(word_dict, desc="embedding...", total=len(word_dict)):
    index = word_dict[word]
    if index != 0:
        if word in glove_stoi:
            word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]
        else:
            random_vector = torch.zeros(word_embedding_dim)
            random_vector.normal_(mean=0, std=0.1)
            word_embedding_vectors[index, :] = random_vector + glove_mean_vector
with open("../../MIND-small/all_concept_word_embedding.npy", 'wb') as word_embedding_f:
    np.save(word_embedding_f, word_embedding_vectors)


