#Workflow

#Download BERT and Wiki corpus

#(For each word)
#Get sentences from Wiki and sample 1000 (or N) of them
#Get clusters of the plot
#(Optional) Make a plot of the clustering

#!/bin/bash

mkdir results/cluster_points
mkdir results/plots

parallel -a words.txt python scripts/senses.py

#FILE="words.txt"
#LINES=$(cat $FILE)
#for LINE in $LINES
#do
#  python scripts/senses.py $LINE
#done
