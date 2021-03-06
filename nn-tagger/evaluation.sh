#!/bin/bash

# make sure correct character encoding is used
LANG=de_DE.UTF-8

printf "# MODEL EVALUATION\n"

# FNN evaluation
printf "\n## N Past Words:\n"
for p in {0..12}; do
    if [ -d "storage/fnn-$p-50-100-1" ]; then
        printf "Model fnn-$p-50-100-1\n"
        python tagger.py --reset -fq
        cp storage/fnn-$p-50-100-1/* saved/
        python tagger.py --evaluate data/evaluation_known.txt -p $p -e 50 -s 100 -n 1 -i
        python tagger.py --evaluate data/evaluation_unknown.txt -p $p -e 50 -s 100 -n 1 -i
    fi
done

printf "\n## Embedding Size:\n"
for e in 1 5 10 25 50 100 150 200 250 300 350 400 450 500 550 600 650 700; do
    if [ -d "storage/fnn-1-$e-100-1" ]; then
        printf "Model fnn-1-$e-100-1\n"
        python tagger.py --reset -fq
        cp storage/fnn-1-$e-100-1/* saved/
        python tagger.py --evaluate data/evaluation_known.txt -p 1 -e $e -s 100 -n 1 -i
        python tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e $e -s 100 -n 1 -i
    fi
done

printf "\n## Hidden Layer Size:\n"
for s in 10 25 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000; do
    if [ -d "storage/fnn-1-50-$s-1" ]; then
        printf "Model fnn-1-50-$s-1\n"
        python tagger.py --reset -fq
        cp storage/fnn-1-50-$s-1/* saved/
        python tagger.py --evaluate data/evaluation_known.txt -p 1 -e 50 -s $s -n 1 -i
        python tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e 50 -s $s -n 1 -i
    fi
done

printf "\n## Training Epochs:\n"
for n in 1 5 10 20 40 60 80 100 120 140; do
    if [ -d "storage/fnn-1-50-100-$n" ]; then
        printf "Model fnn-1-50-100-$n\n"
        python tagger.py --reset -fq
        cp storage/fnn-1-50-100-$n/* saved/
        python tagger.py --evaluate data/evaluation_known.txt -p 1 -e 50 -s 100 -n $n -i
        python tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e 50 -s 100 -n $n -i
    fi
done

printf "\n## Embedding Size II:\n"
for e in 200 250 300 350 400 450 500 550 600; do
    if [ -d "storage/fnn-1-$e-350-1" ]; then
        printf "Model fnn-1-$e-350-1\n"
        python tagger.py --reset -fq
        cp storage/fnn-1-$e-350-1/* saved/
        python tagger.py --evaluate data/evaluation_known.txt -p 1 -e $e -s 350 -n 1 -i
        python tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e $e -s 350 -n 1 -i
    fi
done

printf "\n## Hidden Layer Size II:\n"
for s in 200 250 300 350 400 450 500 550 600; do
    if [ -d "storage/fnn-1-250-$s-1" ]; then
        printf "Model fnn-1-250-$s-1\n"
        python tagger.py --reset -fq
        cp storage/fnn-1-250-$s-1/* saved/
        python tagger.py --evaluate data/evaluation_known.txt -p 1 -e 250 -s $s -n 1 -i
        python tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e 250 -s $s -n 1 -i
    fi
done

printf "\n## Network Size:\n"
for n in {50..500..50}; do
    if [ -d "storage/fnn-1-$n-$n-5" ]; then
        printf "Model fnn-1-$n-$n-5\n"
        python tagger.py --reset -fq
        cp storage/fnn-1-$n-$n-5/* saved/
        python tagger.py --evaluate data/evaluation_known.txt -p 1 -e $n -s $n -n 5 -i
        python tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e $n -s $n -n 5 -i
    fi
done

# RNN evaluation
printf "\n## N Time Steps:\n"
for t in {1..12}; do
    if [ -d "storage/rnn-$t-50-100-1" ]; then
        printf "Model rnn-$t-50-100-1\n"
        python tagger.py --reset -fq
        cp storage/rnn-$t-50-100-1/* saved/
        python tagger.py --evaluate data/evaluation_known.txt -t $t -e 50 -s 50 -n 1 -i
        python tagger.py --evaluate data/evaluation_unknown.txt -t $t -e 50 -s 50 -n 1 -i
    fi
done

printf "\n## Embedding Size:\n"
for e in 1 5 10 25 50 100; do
    if [ -d "storage/rnn-1-$e-100-1" ]; then
        printf "Model rnn-1-$e-100-1\n"
        python tagger.py --reset -fq
        cp storage/rnn-1-$e-100-1/* saved/
        python tagger.py --evaluate data/evaluation_known.txt -t 1 -e $e -s 50 -n 1 -i
        python tagger.py --evaluate data/evaluation_unknown.txt -t 1 -e $e -s 50 -n 1 -i
    fi
done

printf "\n## Hidden Layer Size:\n"
for s in 10 25 50 100 150 200; do
    if [ -d "storage/rnn-1-50-$s-1" ]; then
        printf "Model rnn-1-50-$s-1\n"
        python tagger.py --reset -fq
        cp storage/rnn-1-50-$s-1/* saved/
        python tagger.py --evaluate data/evaluation_known.txt -t 1 -e 50 -s $s -n 1 -i
        python tagger.py --evaluate data/evaluation_unknown.txt -t 1 -e 50 -s $s -n 1 -i
    fi
done

printf "\n## Training Epochs:\n"
for n in 1 5 10 20; do
    if [ -d "storage/rnn-1-50-100-$n" ]; then
        printf "Model rnn-1-50-100-$n\n"
        python tagger.py --reset -fq
        cp storage/rnn-1-50-100-$n/* saved/
        python tagger.py --evaluate data/evaluation_known.txt -t 1 -e 50 -s 50 -n $n -i
        python tagger.py --evaluate data/evaluation_unknown.txt -t 1 -e 50 -s 50 -n $n -i
    fi
done
