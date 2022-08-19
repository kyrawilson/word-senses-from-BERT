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
import argparse
#faulthandler.enable()

#TODO: add sample parameter?
def sample_sents(corpus_path, word, n):
    word = word.strip().lower()
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
def get_context_embeddings(sample, model):
    embed_dict = {}
    embedding = TransformerWordEmbeddings(model)
    for sent in sample:
        sentence = Sentence(sent)
        embedding.embed(sentence)
        for token in sentence:
            if token.text.lower() == word.lower():
                embed_dict[sent] = token.embedding
    return embed_dict

#Reduce dimensions of BERT embeddings for better clustering performance
def reduce_dim(embed_dict, tsne, dim):
    data = []
    sents = []
    for key in embed_dict.keys():
        data.append(embed_dict[key].numpy())
        sents.append(key)
    data = np.array(data)
    sents = np.array(sents)
    if tsne:
        reduce = TSNE(n_components=dim, random_state=0)
        projections = reduce.fit_transform(data)
    else:
        reduce = PCA(n_components=dim, random_state=0)
        projections = reduce.fit_transform(data)
    return sents, projections

#TODO: should also save all words with senses somehow
def make_clusters(sents, projections, minsize):
    if minsize is False:
        minsize =  int(len(sents)*0.01)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, minsize]))
    cluster_labels = clusterer.fit_predict(projections)
    projections_t = np.transpose(projections)
    data = np.stack((sents, cluster_labels, projections_t[0], projections_t[1])).T
    df = pd.DataFrame(data, columns=['sentence', 'sense_labels', 'dim1', 'dim2'])
    clusters = int(df['sense_labels'].max()) + 1
    if -1 in df['sense_labels']:
        clusters -= 1
    df.to_csv(f'results/sense_points/{word}_{clusters}.csv', index=False)
    return clusters

#Make a plotly plot of the clusters for a given word
def plots(word, clusters, type):
    df = pd.read_csv(f'results/{type}_points/{word}_{clusters}.csv')
    df[f"{type}_labels"] = df[f"{type}_labels"].astype(str)

    fig = px.scatter(df, x='dim1', y='dim2',
    color=f"{type}_labels", hover_data=['sentence'], title=f"{word} {type} clusters")
    fig.write_html(f"results/{type}_plots/{word}_{clusters}.html")

def meanings(word, clusters, minsize):
    df = pd.read_csv(f'results/sense_points/{word}_{clusters}.csv')
    if minsize is False:
        minsize = int(len(df['sentence'])*0.01)
    projections = df[['dim1', 'dim2']]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, minsize])).fit(projections)
    exemplars_2d = clusterer.exemplars_

    if exemplars_2d:
        min_size = min([len(sublist) for sublist in exemplars_2d])

        exemplars_2d = [item for sublist in exemplars_2d for item in sublist]
        exemplars_2d = np.array(exemplars_2d)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, allow_single_cluster=True)
        labels = clusterer.fit_predict(exemplars_2d)

        projections_t = np.transpose(exemplars_2d)
        data = np.stack((projections_t[0], projections_t[1], labels)).T
        df2 = pd.DataFrame(data, columns=["dim1", "dim2", "meaning_labels"])
        x = df.merge(df2, on=["dim1", "dim2"], how="inner")
        meaning_clusters = int(x['meaning_labels'].max()) + 1
        if -1 in x['meaning_labels']:
            meaning_clusters -= 1
        x[['sentence', 'meaning_labels', 'dim1', 'dim2']].to_csv(f'results/meaning_points/{word}_{int(meaning_clusters)}.csv', index=False)

    else:
        print(f"Could not calculate number of meanings for {word}; insufficient sense clusters.")
        clusters = 0

    if clusters > meaning_clusters:
        clusters = meaning_clusters

    return int(clusters)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("word", help="the word to get the number of senses for")
    parser.add_argument("corpus", help="path to the corpus file")
    parser.add_argument("--sample", type=int, help="number of sentences to sample from corpus", default=1000)
    parser.add_argument("--plots", type=bool, help="make plots of sampled sentences and clusters", default=False)
    parser.add_argument("--tsne", type=bool, help="use tSNE for dimensionality reduction (default PCA)", default=False)
    parser.add_argument("--meanings", type=bool, help="calculate number of meanings as well", default=False)
    parser.add_argument("--model", help="name of contextual embedding model to use", default='bert-base-cased')
    parser.add_argument("--minsize", type=int, help="minimum size for formation of sense clusters", default=False)
    parser.add_argument("--dim", type=int, help="size of reduced dimension embeddings", default=2)
    args = parser.parse_args()

    corpus_path = args.corpus
    word = args.word
    sample = sample_sents(corpus_path, word, args.sample)
    embed_dict = get_context_embeddings(sample, args.model)
    sents, projections = reduce_dim(embed_dict, args.tsne,  args.dim)
    sense_clusters = make_clusters(sents, projections, args.minsize)
    if args.plots:
        plots(word, sense_clusters, 'sense')
    if args.meanings:
        meaning_clusters = meanings(word, sense_clusters, args.minsize)
    if args.meanings and args.plots and meaning_clusters != 0:
        plots(word, meaning_clusters, 'meaning')
