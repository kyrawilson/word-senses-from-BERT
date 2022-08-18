# word-senses-from-CWE

Calculate number of senses and meanings of words using contextualized word embeddings

<h2>Background</h2>

Calculating the number of senses and meanings a word has is usually done by consulting a dictionary or other lexicographic resource. This tool enables a hands-off computation of the number of senses and meanings using contextualized word embeddings. We sample occurrences of the words in a corpus, and then use BERT to derive embeddings for each of these words in context. The HDBSCAN clustering algorithm is then applied to a dimension-reduced version of the embeddings, and the number of senses is equal to the number of clusters identified. Calculation of meanings follows much the same procedure, except exemplar points of the sense clusters are used rather than the complete set of contexts. A full description of the algorithm will be published shortly. 

<h2>Getting Started</h2>

After cloning the repository, add your corpus and list of words to the ```data/``` folder. The corpus should be one sentence per line, and the list of words should have one word per line. See ```data/sample_corpus.txt``` and ```data/sample_words.txt``` as an example. 

Additionally, you may find it useful to create a new conda environment using the provided ```environment.yml``` file.

<h2>Running the Code</h2>

Before running the code, change the path to the list of words and corpus in ```scripts/pipeline.sh``` as follows:
    
```
parallel --bar -a your/word/list.txt python scripts/senses.py {} your/corpus/file.txt
```

Change parameters using optional flags on the same line: 

```
parallel --bar -a your/word/list.txt python scripts/senses.py {} your/corpus/file.txt --meanings True --plots True
```

See all parameters by running

```
python scripts/senses.py -h
```

Finally, run the code. GNU Parallel is used to speed up computation time by default. 

```
./scripts/pipeline.sh
```

<h2>Results</h2>

Output is saved in ```results/```. A complete list of words and their number of senses and meanings can be found in ```results/senses.txt``` and ```results/meanings.txt```.

If you would like a more detailed look at individual words, the complele lists of sampled contexts, their clusters, and their two-dimensional embeddings are found in ```results/sense_points/``` and ```results/meaning_points/```.

The same information presented graphically can be found in the ```results/sense_plots/``` and ```results/meaning_plots/``` folders.

<h2>Acknowledgements</h2>
This repository was created with the help of The Good Research Code Handbook(https://goodresearch.dev/index.html).
