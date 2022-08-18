#Workflow

#Download BERT and Wiki corpus

#(For each word)
#Get sentences from Wiki and sample 1000 (or N) of them
#Get clusters of the plot
#(Optional) Make a plot of the clustering

#!/bin/bash
mkdir results/sense_points
mkdir results/meaning_points
mkdir results/sense_plots
mkdir results/meaning_plots

parallel --bar -a data/sample_words.txt python scripts/senses.py {} data/sample_corpus.txt --meanings True --plots True

# FILE="data/sample_words.txt"
# LINES=$(cat $FILE)
# for LINE in $LINES
# do
#  python scripts/senses.py $LINE data/sample_corpus.txt
# done

ls results/sense_points | sed -e 's/_/,/g;s/.csv//g' > results/senses.csv
ls results/meaning_points | sed -e 's/_/,/g;s/.csv//g' > results/meanings.csv
