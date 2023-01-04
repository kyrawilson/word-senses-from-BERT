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
import pickle
#faulthandler.enable()

#TODO: add sample parameter?

def preprocess(corpus_path, word_path, n):
    sentences = {}
    #Get all sentences from corpus where word appears a single time
    with open(corpus_path, 'r') as file:
        with open(word_path, 'r') as file2:
            for line in file:
                count = 0
                for line in file2:
                    word = line.strip().lower()
                    if word not in sentences.keys():
                        sentences[word] = []
                    if len(re.findall(rf'\b{word}\b', line.lower())) == 1:
                        sentences[word].append(count)
                count += 1
    #Sample from list of sentences
    for key in sentences.keys():
        if len(sentences[key]) > n:
            sample = random.sample(sentences[key], n)
            sentences[key] = sample
    else:
        print(f"Warning: {word} had only {len(sents)} occurrences in corpus.")
        #sample = sents
    pkl_path = f'{corpus_path[:-4]}_preprocessed.pkl'
    output = open(pkl_path, 'wb')
    pickle.dump(sentences, output)
    output.close()
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("word", help="path to the word list")
    parser.add_argument("corpus", help="path to the corpus file")
    parser.add_argument("--sample", type=int, help="number of sentences to sample from corpus", default=1000)
    args = parser.parse_args()

    corpus_path = args.corpus
    word_path = args.word
    p = preprocess(corpus_path, word_path, args.sample)
