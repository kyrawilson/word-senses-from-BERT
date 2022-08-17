import nltk
from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwordsent
import time
from sys import getsizeof
import os
import re
import sys
from datetime import datetime
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
import pandas as pd
import random
import pickle
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np
import hdbscan
from scipy.stats import entropy

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
np.set_printoptions(threshold=np.inf)


def sents():
    stim_sents = {}
    with open('FINAL/wiki_sentences.txt', 'r') as file:
        with open('/scratch/kew6086/NV/Stimuli/stimuli.txt', 'r') as file2:
            for word in file2:
                stim_sents[word] = []
            for line in file:
                for word in stim_sents.keys():
                    if len(re.findall(rf'\b{word}\b', line)) == 1:
                        stim_sents[word].append(line)
            for word in stim_sents.keys():
                with open(f'Stim_sentences/untagged/{word}.txt', 'w') as file3:
                    print(f'{word}: {len(stim_sents[word])}')
                    file3.writelines(stim_sents[word])

#File for NV stimuli: f'Stim_sentences/untagged/{word}.txt' 
#File for Rodd2: 'Rodd/Rodd2/sentences/{word}.txt'
def sents2(word):
    word = word.strip()
    stim_sents = []
    temp = ''
    count = 0
    with open('FINAL/wiki_sentences.txt', 'r') as file:
        with open('out.txt', 'w') as file2:
            for line in file:
                if len(re.findall(rf'\b{word}\b', line.lower())) == 1:
                    stim_sents.append(line)
                temp += line
                if getsizeof(temp) >= 1000000000:
                    temp = ''
                    count += 1
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    file2.write(f'{current_time} {count}\n')
        with open(f'Rodd/Rodd1/sentences/{word}.txt', 'w') as file3:
            print(f'{word}: {len(stim_sents)}')
            file3.writelines(stim_sents)

#File for NV stimuli: f'Stim_sentences/untagged/{word}.txt'                
def tag(word):
    with open(f'Stim_sentences/untagged/{word}.txt') as file:
        sentences = file.readlines()
        tokenized_sents = [nltk.word_tokenize(s) for s in sentences]
    
        idx = []
        for sent in tokenized_sents:
            sent_lower = [x.lower() for x in sent]
            try:
                ind = sent_lower.index(word)
                idx.append(ind)
            except ValueError:
                idx.append("NA")
            
        tagged_sents = nltk.pos_tag_sents(tokenized_sents)
        
        with open(f'Stim_sentences/tagged_N/{word}_N.txt', 'w') as fileN:
            with open(f'Stim_sentences/tagged_V/{word}_V.txt', 'w') as fileV:
                count = 0
                for index, sentence in zip(idx,tagged_sents):
                    if index == 'NA':
                        pass
                    else:
                        if sentence[index][1].startswith('N'):
                            fileN.write(f'{sentences[count]}')
                        elif sentence[index][1].startswith('V'):
                            fileV.write(f'{sentences[count]}')
                    count += 1

#Path for NV: 'Stim_sentences/tagged_V/COUNT_tagged_V.csv', 'Stim_sentences/tagged_V/', 'Stim_sentences/tagged_V/' + f,
#Path for Rodd2: 'Rodd/Rodd2/COUNT.csv', 'Rodd/Rodd2/sentences/', 'Rodd/Rodd2/sentences/'
def count():
    with open('Rodd/Rodd1/COUNT.csv', 'w') as write:
        for f in os.listdir('Rodd/Rodd1/sentences/'):
            with open('Rodd/Rodd1/sentences/' + f) as file:
                word = f.split('.')[0]
                count = 0
                for line in file:
                    count += 1
            write.write(f'{word},{count}\n')

#Path for NV: f'Stim_sentences/COUNT_{folder}.csv', 'Stim_sentences/{path}.txt', f'embeddings/{folder}/{word}.pickle'   
#Path for Rodd2: f'Rodd/Rodd2/COUNT.csv', f'Rodd/Rodd2/sentences/{word}.txt', f'Rodd/Rodd2/embeddings/{word}.pickle'
def sample(path):
    word = path.strip()
    folder = ''
    if "/" in path:
        word = path.split('/')[1].strip()
        folder = path.split('/')[0]
    counts = pd.read_csv(f'Rodd/Rodd1/COUNT.csv', index_col=0, header=None, squeeze=True).to_dict()
    #counts = {k.split("_")[0]:v for k,v in counts.items()}
    count = counts[word]
    sentences = []
    
    with open(f'Rodd/Rodd1/sentences/{word}.txt') as file:
        if count > 1000:
            #sample 1000 random lines
            l = random.sample(range(count), 1000)
            l.sort()
            #print(l)
            cur = 0
            for line in file:
                if cur == l[0]:
                    sentences.append(line)
                    del l[0]
                    if len(l) == 0:
                        break
                    cur += 1
                else:
                    cur += 1
        else:
            sentences = file.readlines()
    
    embed_dict = {}
    embedding = TransformerWordEmbeddings('bert-base-cased')
    for sent in sentences:
        sentence = Sentence(sent)
        embedding.embed(sentence)
        for token in sentence:
            if token.text.lower() == word.split("_")[0]:
                embed_dict[sent] = token.embedding
                
    with open(f'Rodd/Rodd1/embeddings/{word}.pickle', 'wb') as handle:
        pickle.dump(embed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def plot(word):
    word = word.strip()
    with open (f'embeddings/tagged_N/{word}_N.pickle', "rb") as N_file:
        with open (f'embeddings/tagged_V/{word}_V.pickle', "rb") as V_file:
            N = pickle.load(N_file)
            V = pickle.load(V_file)
            
            N_embed = []
            V_embed = []
            N_sent = []
            V_sent = []
            N_color = []
            V_color = []
            for key in N.keys():
                N_embed.append(N[key].numpy())
                N_sent.append(key)
                N_color.append("N")
            for key in V.keys():
                V_embed.append(V[key].numpy())
                V_sent.append(key)
                V_color.append("V")
                
            all_embed = np.array(N_embed + V_embed)
            all_sent = np.array(N_sent + V_sent)
            all_color = np.array(N_color + V_color)
            
            tsne = TSNE(n_components=2, random_state=0)
            projections = tsne.fit_transform(all_embed)

            fig = px.scatter(
            projections, x=0, y=1,
            color=all_color, hover_data=[all_sent]
            )
            fig.write_html(f"plots/{word}.html")

#Plotting should be same locations, difference is whether clusters are based off high dimensional or low dimensional vectors
#NV stimuli: f'embeddings/untagged/{word}.pickle', f"clustering/untagged/plots/{word}-HD-{num}.html", f"clustering/untagged/plots/{word}-2D-{num2}.html", f'clustering/untagged/v2_NumClust/{word}.txt'
#Rodd2: f'Rodd/Rodd2/embeddings/{word}.pickle', f"Rodd/Rodd2/clustering/plots/HD/{word}-HD_{num}.html", f"Rodd/Rodd2/clustering/plots/2D/{word}-2D_{num2}.html"
#Rodd1: f'Rodd/Rodd1/embeddings/{word}.pickle', f"Rodd/Rodd1/clustering/plots/HD/{word}-HD_{num}.html", "Rodd/Rodd1/clustering/plots/2D/{word}-2D_{num2}.html", f'Rodd/Rodd1/clustering/clusters/v2_NumClust/{word}.txt'
def untagged_clustering(word):
    word = word.strip()
    with open (f'embeddings/untagged/{word}.pickle', "rb") as file:
        embed_dict = pickle.load(file)
        
        data = []
        sent = []
        for key in embed_dict.keys():
            data.append(embed_dict[key].numpy())
            sent.append(key)
        
        data = np.array(data)
        sent = np.array(sent)
        
        #Make clusters
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sent)*0.01)]))
        cluster_labels = clusterer.fit_predict(data)
        
        #Reduce dimensions of data for plotting purposes
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(data)
        
        num = np.size(np.unique(cluster_labels))
        
        fig = px.scatter(
        projections, x=0, y=1,
        color=cluster_labels, hover_data=[sent]
        )
        fig.write_html(f"clustering/untagged/plots/{word}-HD-{num}.html")
        
        #Cluster reduced dimensions
        cluster_labels2 = clusterer.fit_predict(projections)
        num2 = np.size(np.unique(cluster_labels2))
        
        fig = px.scatter(
        projections, x=0, y=1,
        color=cluster_labels2, hover_data=[sent]
        )
        fig.write_html(f"clustering/untagged/plots/{word}-2D-{num2}.html")
        
        if -1 in np.unique(cluster_labels):
            num -= 1
        if -1 in np.unique(cluster_labels2):
            num2 -= 1
        
        out = f'{word}, {num}, {num2}'
    
        with open (f'clustering/untagged/v2_NumClust/{word}.txt', "w") as writeFile:
            writeFile.write(out)
        
#NV stimuli: f'embeddings/untagged/{word}.pickle', f'clustering/untagged/2D/{word}.csv'
#Rodd2: f'Rodd/Rodd2/embeddings/untagged/{word}.pickle', f'Rodd/Rodd2/clustering/untagged/2D/{word}.csv'
#NEED TO CHANGE FILES FOR RODD REPLICATION
def get_clusters_2d(word):
    word = word.strip()
    with open (f'Rodd/Rodd1/embeddings/{word}.pickle', "rb") as file:
        embed_dict = pickle.load(file)
        
        data = []
        sent = []
        for key in embed_dict.keys():
            data.append(embed_dict[key].numpy())
            sent.append(key)
        
        data = np.array(data)
        sent = np.array(sent)
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sent)*0.01)]))
        #cluster_labels = clusterer.fit_predict(data)
        
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(data)
        cluster_labels = clusterer.fit_predict(projections)
        
        labels = ['sent', 'cluster', 'dim1', 'dim2']
        projections = np.transpose(projections)
        #print(projections)
        data = np.stack((sent, cluster_labels, projections[0], projections[1])).T
        print(data)
        df = pd.DataFrame(data, columns=labels)
        df.to_csv(f'Rodd/Rodd1/clustering/untagged/2D/{word}.csv', index=False)
        
def get_tagged_clusters_2d(word):
    data_all = []
    for tag in ["N", "V"]:
        word1 = f"{word}_{tag}"
        folder = f"tagged_{tag}"
        with open (f'embeddings/{folder}/{word1}.pickle', "rb") as file:
            embed_dict = pickle.load(file)
        
            data = []
            sent = []
            for key in embed_dict.keys():
                data.append(embed_dict[key].numpy())
                sent.append(key)
            
            data = np.array(data)
            sent = np.array(sent)
            print(len(data))
            
            if len(data) >= 1:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sent)*0.01)]))
                cluster_labels = clusterer.fit_predict(data)
                cluster_labels_adj = np.array([str(i) + f"_{tag}" for i in cluster_labels])
        
                tsne = TSNE(n_components=2, random_state=0)
                projections = tsne.fit_transform(data)
        
                projections = np.transpose(projections)
                #print(projections)
                data = np.stack((sent, cluster_labels_adj, projections[0], projections[1])).T
                data_all.append(data)
            
    d = np.concatenate(data_all)
    labels = ['sent', 'cluster', 'dim1', 'dim2']
    df = pd.DataFrame(d, columns=labels)
    df.to_csv(f'clustering/tagged/2D/{word}.csv', index=False)

#NV stimuli: f'clustering/{folder}/2D/', f'clustering/{folder}/2D/{f}', f'clustering/{folder}/{folder}_2D_dist.csv'
#Rodd replication: f'Rodd/Rodd2/clustering/2D/', f'Rodd/Rodd2/clustering/2D/{f}', f'Rodd/Rodd2/clustering/{folder}_2D_dist.csv'
def cluster_dist_2d(folder):
    data = []
    
    for f in os.listdir(f'Rodd/Rodd1/clustering/{folder}/2D/'):
        df = pd.read_csv(f'Rodd/Rodd1/clustering/{folder}/2D/{f}')
        word = f.split('.')[0].strip()
    
        all_center = np.array([df['dim1'].mean(), df['dim2'].mean()])
        
        clust_center = df.groupby('cluster').mean()
        clust_center['dist'] = np.linalg.norm(clust_center[['dim1', 'dim2']].sub(all_center), axis=1)
        
        data.append([word, len(clust_center), clust_center['dist'].mean()])
        
    #save results
    df_write = pd.DataFrame(data, columns =['word',f'{folder}_2D', f'{folder}_2D_dist'])
    df_write.to_csv(f'Rodd/Rodd1/clustering/{folder}/{folder}_2D_dist.csv', index=False)
    
def tree_plots(word):
    word = word.strip()
    with open (f'Rodd/embeddings/{word}.pickle', "rb") as file:
        embed_dict = pickle.load(file)
        
        data = []
        sent = []
        for key in embed_dict.keys():
            data.append(embed_dict[key].numpy())
            sent.append(key)
        
        data = np.array(data)
        sent = np.array(sent)
        
        #Make clusters
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sent)*0.01)])).fit(data)
        plt = clusterer.condensed_tree_.plot(select_clusters=True)
        plt.figure.savefig(f"Rodd/clustering/plots/trees/HD/{word}.png")
        
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(data)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sent)*0.01)])).fit(projections)
        plt = clusterer.condensed_tree_.plot(select_clusters=True)
        plt.figure.savefig(f"Rodd/clustering/plots/trees/2D/{word}.png")

#Rodd2 files: f'Rodd/Rodd2/embeddings/{word}.pickle', f'Rodd/Rodd2/clustering/persistence/{word}.txt'
def persistence(word):
    
    word = word.strip()
    with open (f'Rodd/Rodd1/embeddings/{word}.pickle', "rb") as file:
        embed_dict = pickle.load(file)
        
        data = []
        sent = []
        for key in embed_dict.keys():
            data.append(embed_dict[key].numpy())
            sent.append(key)
        
        data = np.array(data)
        sent = np.array(sent)
        
        #Make clusters
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sent)*0.01)])).fit(data)
        persist_hd = clusterer.cluster_persistence_
        print(persist_hd)
        
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(data)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sent)*0.01)])).fit(projections)
        persist_2d = clusterer.cluster_persistence_

        out = f'{word.strip()}, {np.average(persist_hd)}, {np.average(persist_2d)}'
    
        with open (f'Rodd/Rodd1/clustering/persistence/{word}.txt', "w") as writeFile:
            writeFile.write(out)

#Rodd2 files: f'Rodd/Rodd2/embeddings/{word}.pickle', f'Rodd/Rodd2/clustering/exemplars/{word}.txt'
def exemplars(word):
    
    word = word.strip()
    with open (f'Rodd/Rodd1/embeddings/{word}.pickle', "rb") as file:
        embed_dict = pickle.load(file)
        
        data = []
        sent = []
        for key in embed_dict.keys():
            data.append(embed_dict[key].numpy())
            sent.append(key)
        
        data = np.array(data)
        sent = np.array(sent)
        
        #Make clusters
        #change so that 1 is a valid cluster size
        #change to have more empirically motivated min cluster size
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sent)*0.01)])).fit(data)
        exemplars_hd = clusterer.exemplars_
        print("orig HD size: ", len(exemplars_hd)) 
        exemplars_hd = [item for sublist in exemplars_hd for item in sublist]
        exemplars_hd = np.array(exemplars_hd)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, allow_single_cluster=True)
        labels = clusterer.fit_predict(exemplars_hd)
        hd_size = np.unique(labels).size
        print("exemp HD size: ", hd_size)
        
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(data)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sent)*0.01)])).fit(projections)
        exemplars_2d = clusterer.exemplars_
        print("orig 2D size: ", len(exemplars_2d))
        exemplars_2d = [item for sublist in exemplars_2d for item in sublist]
        exemplars_2d = np.array(exemplars_2d)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, allow_single_cluster=True)
        labels = clusterer.fit_predict(exemplars_2d)
        td_size = np.unique(labels).size
        print("exemp 2D size: ", td_size)

        #out = f'{word.strip()}, {np.average(exemplars_hd)}, {np.average(exemplars_2d)}'
        out = f'{word}, {hd_size}, {td_size}'
    
        with open (f'Rodd/Rodd1/clustering/exemplars/edit/{word}.txt', "w") as writeFile:
            writeFile.write(out)

#Rodd1 exemplars: "Rodd1_exemplars.csv", ['Word', 'Exemplar_HD', 'Exemplar_2D'] 
#NV senses: path=clustering/untagged/v2_NumClust, 'clustering/untagged/BERT_senses.csv', ['Word', 'BERT_senses_HD', 'BERT_senses_2D']
#NV meanings: path=clustering/untagged/exemplars/meanings, 'clustering/untagged/exemplars/BERT_meanings.csv', ['Word', 'BERT_meanings']
#NV tagged: path=clustering/tagged/exemplars/meanings/N, 'clustering/tagged/exemplars/meanings/BERT_meanings_N.csv', ['Word', 'BERT_meanings_N', 'BERT_senses_N']
def stitch(path):
    all = []
    files = os.listdir(path)
    for filename in files:
        data = pd.read_csv(f'{path}/{filename}', header=None)
        all.append(data)
    df = pd.concat(all)
    df.to_csv('clustering/untagged/exemplars/BERT_meanings.csv', index=False, header=['Word', 'BERT_meanings', 'BERT_senses', 'BERT_meaning_entropy'])
    
def entropy_old(word):
    word = word.strip()
    with open (f'Rodd/Rodd1/embeddings/{word}.pickle', "rb") as file:
        embed_dict = pickle.load(file)
        
        data = []
        sent = []
        for key in embed_dict.keys():
            data.append(embed_dict[key].numpy())
            sent.append(key)
        
        data = np.array(data)
        sent = np.array(sent)
        
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(data)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sent)*0.01)])).fit(projections)
        prob = clusterer.probabilities_
        labels = clusterer.labels_
        temp = np.stack((labels, prob), axis=-1)
        df = pd.DataFrame(temp, columns=['Label', 'Probability'])
        mean = df.groupby('Label').mean()
        print(word)
        print(mean)
        
        
#Rodd2 files: f'Rodd/Rodd2/embeddings/{word}.pickle', f'Rodd/Rodd2/clustering/exemplars/{word}.txt'
#Rodd1 files: f'Rodd/Rodd1/embeddings/{word}.pickle', f"Rodd/Rodd1/clustering/exemplars/edit/plots/{word}.html", f'Rodd/Rodd1/clustering/exemplars/edit/v2/{word}.txt'
#NV files: f'embeddings/untagged/{word}.pickle', f'/clustering/untagged/exemplars/plots/{word}.html", f'/clustering/untagged/exemplars/meanings/{word}.txt'
#Tagged files: f'embeddings/tagged_V/{word}_V.pickle', f'clustering/tagged/exemplars/plots/{word}_V.html', f'clustering/tagged/exemplars/meanings/V/{word}_V.txt'
def exemplars_edit(word):
    
    word = word.strip()
    with open ( f'embeddings/untagged/{word}.pickle', "rb") as file:
        embed_dict = pickle.load(file)
        
        if embed_dict:
        
            data = []
            sent = []
            for key in embed_dict.keys():
                data.append(embed_dict[key].numpy())
                sent.append(key)
            
            data = np.array(data)
            sent = np.array(sent)
            
            #print(data)
            
            tsne = TSNE(n_components=2, random_state=0)
            projections = tsne.fit_transform(data)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max([2, int(len(sent)*0.01)])).fit(projections)
            exemplars_2d = clusterer.exemplars_
            symbols = clusterer.labels_
            
            clust_size = len(np.unique(symbols))
            if -1 in np.unique(symbols):
                clust_size -= 1
            
            if exemplars_2d:
                exemplars_dict = {}
                #for i in range(len(exemplars_2d)):
                    #for item in exemplars_2d[i]:
                        #print(item)
                        #exemplars_dict[str(item.tolist())] = i
                orig_size = len(exemplars_2d)
                
                #print(exemplars_2d)
                #print([len(sublist) for sublist in exemplars_2d])
                min_size = min([len(sublist) for sublist in exemplars_2d])
                
                exemplars_2d = [item for sublist in exemplars_2d for item in sublist]
                exemplars_2d = np.array(exemplars_2d)
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, allow_single_cluster=True)
                labels = clusterer.fit_predict(exemplars_2d)
                
                #Calculate entropy
                (unique_labels, counts) = np.unique(labels, return_counts=True)
                print(unique_labels, counts)
                if -1 in unique_labels:
                    counts = counts[1:]
                meaning_entropy = entropy(counts, base=2)
                
                
                for i in range(len(exemplars_2d)):
                    if labels[i] != -1:
                        exemplars_dict[str(exemplars_2d[i].tolist())] = labels[i]
                    else:
                        exemplars_dict[str(exemplars_2d[i].tolist())] = -2
                        
                    
                try:
                    #Make np array with vals indicating whether the point is an exemplar or not
                    exemplar_size = np.array([2 if pt in exemplars_2d else .5 for pt in projections])
                    exemplar_color = [str(exemplars_dict[str(pt.tolist())]) if pt in exemplars_2d else '-2' for pt in projections]
                    
                    fig = px.scatter(
                    projections, x=0, y=1,
                    symbol=symbols, hover_data=[sent], size=exemplar_size, color=exemplar_color, title=word,
                    symbol_sequence = ["circle", "square", "diamond", "cross", "x", 
                    "triangle-up", "triangle-down", "triangle-left", "triangle-right", "pentagon",
                    "octagon", "star", "hourglass", "hexagon", "circle-cross",
                    "bowtie", "diamond-x"],
                    color_discrete_sequence = ["#708090","#800000", "#FF6347", "#FFA07A", "#FFFF00", "#F0E68C",
                    "#ADFF2F", "#40E0D0", "#3CB371", "#008080", "#4169E1",
                    "#000080", "#87CEFA", "#4B0082", "#9370DB", "#DDA0DD", "#A0522D"
                    ]
                    )
                    fig.write_html(f'clustering/untagged/exemplars/plots/{word}.html')
                
                except KeyError:
                    print("KeyError in plotting.")
                
                td_size = np.unique(labels).size
                #Remove noise points
                if -1 in np.unique(labels):
                    td_size -= 1
                if td_size > orig_size:
                    td_size = orig_size
                print("exemp 2D size: ", td_size)
            
            else:
                td_size = 1
        
        else:
            td_size = 0
            clust_size = 0
    
        #out = f'{word.strip()}, {np.average(exemplars_hd)}, {np.average(exemplars_2d)}'
        #out = f'{word}, {hd_size}, {td_size}'
        out = f'{word}, {td_size}, {clust_size}, {meaning_entropy}'
    
        with open (f'clustering/untagged/exemplars/meanings/{word}.txt', "w") as writeFile:
            writeFile.write(out)
    
    
    

                
#sents2(sys.argv[1])
#tag(sys.argv[1])
#count()
#sample(sys.argv[1])
#plot(sys.argv[1])
#untagged_clustering(sys.argv[1])
#get_clusters_2d(sys.argv[1])
#cluster_dist_2d(sys.argv[1])
#get_tagged_clusters_2d(sys.argv[1])
#cluster_dist_2d(sys.argv[1])  
#tree_plots(sys.argv[1])
#persistence(sys.argv[1])
#exemplars(sys.argv[1])
#entropy(sys.argv[1])
#exemplars_edit(sys.argv[1])
stitch(sys.argv[1])
#meaning_entropy(sys.argv[1])
    
                    



