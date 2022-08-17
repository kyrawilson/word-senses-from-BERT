import os
import re
import sys
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
import pandas as pd
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import hdbscan
import faulthandler

#faulthandler.enable()

#TODO: add sample parameter?
def sample_sents(corpus_path, word, n):
    word = word.strip()
    sents = []
    #Get all sentences from corpus where word appears a single time
    with open(corpus_path, 'r') as file:
        for line in file:
            if len(re.findall(rf'\b{word}\b', line.lower())) == 1:
                sents.append(line)
    #Sample from list of sentences
    if len(sents) > n:
        sample = random.sample(sents, n)
    else:
        print(f"Warning: {word} had only {len(sents)} occurrences in corpus.")
        sample = sents
    return sample

#TODO: add model name parameter
def get_context_embeddings(sample):
    embed_dict = {}
    embedding = TransformerWordEmbeddings('bert-base-cased')
    for sent in sample:
        sentence = Sentence(sent)
        embedding.embed(sentence)
        for token in sentence:
            if token.text.lower() == word:
                embed_dict[sent] = token.embedding
    return embed_dict

#Reduce dimensions of BERT embeddings for better clustering performance
def reduce_dim(embed_dict):
    data = []
    sents = []
    for key in embed_dict.keys():
        data.append(embed_dict[key].numpy())
        sents.append(key)
    data = np.array(data)
    sents = np.array(sents)
    tsne = PCA(n_components=2, random_state=0)
    projections = tsne.fit_transform(data)
    return sents, projections

#TODO: should also save all words with senses somehow
def make_clusters(sents, projections):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sents)*0.01)]))
    cluster_labels = clusterer.fit_predict(projections)
    projections_t = np.transpose(projections)
    data = np.stack((sents, cluster_labels, projections_t[0], projections_t[1])).T
    df = pd.DataFrame(data, columns=['sentence', 'cluster_label', 'dim1', 'dim2'])
    df.to_csv(f'results/cluster_points/{word}.csv', index=False)

#Make a plotly plot of the clusters for a given word
def plot_senses(word):
    df = pd.read_csv(f'results/cluster_points/{word}.csv')
    df["cluster_label"] = df["cluster_label"].astype(str)

    fig = px.scatter(df, x='dim1', y='dim2',
    color='cluster_label', hover_data=['sentence'], title=word
    )
    fig.write_html(f"results/plots/{word}.html")

corpus_path = 'data/sample_corpus.txt'
word = sys.argv[1]
print(f"Calculating senses for '{word}'...")
sample = sample_sents(corpus_path, word, 1000)
embed_dict = get_context_embeddings(sample)
sents, projections = reduce_dim(embed_dict)
make_clusters(sents, projections)
plot_senses(word)
