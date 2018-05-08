#!/bin/bash

# make sure correct character encoding is used
LANG=de_DE.UTF-8

printf "# MODEL EVALUATION\n"

printf "\n## N Past Words:\n"
for p in {0..12}; do
    if [ -d "storage/fnn-$p-50-100-1" ]; then
        printf "Model fnn-$p-50-100-1\n"
        python3.6 tagger.py --reset -fq
        cp storage/fnn-$p-50-100-1/* saved/
        python3.6 tagger.py --evaluate data/evaluation_known.txt -p $p -e 50 -s 100 -n 1 -i
        python3.6 tagger.py --evaluate data/evaluation_unknown.txt -p $p -e 50 -s 100 -n 1 -i
    fi
done

# printf "\n## Embedding Size:\n"
# for e in 1 5 10 25 50 100 150 200 250 300 350; do
#     if [ -d "storage/fnn-1-$e-100-20" ]; then
#         printf "Model fnn-1-$e-100-20\n"
#         python3.6 tagger.py --reset -fq
#         cp storage/fnn-1-$e-100-20/* saved/
#         python3.6 tagger.py --evaluate data/evaluation_known.txt -p 1 -e $e -s 100 -n 20 -i
#         python3.6 tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e $e -s 100 -n 20 -i
#     fi
# done

# printf "\n## Hidden Layer Size:\n"
# for s in 10 25 50 100 150 200 250 300 350 400 450 500 550 600; do
#     if [ -d "storage/fnn-1-50-$s-20" ]; then
#         printf "Model fnn-1-50-$s-20\n"
#         python3.6 tagger.py --reset -fq
#         cp storage/fnn-1-50-$s-20/* saved/
#         python3.6 tagger.py --evaluate data/evaluation_known.txt -p 1 -e 50 -s $s -n 20 -i
#         python3.6 tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e 50 -s $s -n 20 -i
#     fi
# done

# printf "\n## Training Epochs:\n"
# for n in 1 5 10 20 40 60 80 100 120 140; do
#     if [ -d "storage/fnn-1-50-100-$n" ]; then
#         printf "Model fnn-1-50-100-$n\n"
#         python3.6 tagger.py --reset -fq
#         cp storage/fnn-1-50-100-$n/* saved/
#         python3.6 tagger.py --evaluate data/evaluation_known.txt -p 1 -e 50 -s 100 -n $n -i
#         python3.6 tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e 50 -s 100 -n $n -i
#     fi
# done