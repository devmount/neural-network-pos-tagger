#!/bin/bash

# make sure correct character encoding is used
LANG=de_DE.UTF-8

printf "# MODEL TRAINING\n"
mkdir storage

printf "\n## N Past Words:\n"
for p in {2..10}; do
    if [ ! -d "storage/fnn-$p-50-100-20" ]; then
        mkdir storage/fnn-$p-50-100-20
        { time python3.6 tagger.py --train data/training.corpus -p $p -e 50 -s 100 -n 20 ; } > storage/fnn-$p-50-100-20/training.txt 2>&1
        { python3.6 tagger.py --evaluate data/evaluation_known.txt -p $p -e 50 -s 100 -n 20 ; } > storage/fnn-$p-50-100-20/result_known.txt 2>&1
        { python3.6 tagger.py --evaluate data/evaluation_unknown.txt -p $p -e 50 -s 100 -n 20 ; } > storage/fnn-$p-50-100-20/result_unknown.txt 2>&1
        cp saved/* storage/fnn-$p-50-100-20/
        printf "Model with %i past words is saved and evaluated.\n" $p
        python3.6 tagger.py --reset -fq
    fi
done

printf "\n## Embedding Size:\n"
for e in {50..200..50}; do
    if [ ! -d "storage/fnn-1-$e-100-20" ]; then
        mkdir storage/fnn-1-$e-100-20
        { time python3.6 tagger.py --train data/training.corpus -p 1 -e $e -s 100 -n 20 ; } > storage/fnn-1-$e-100-20/training.txt 2>&1
        { python3.6 tagger.py --evaluate data/evaluation_known.txt -p 1 -e $e -s 100 -n 20 ; } > storage/fnn-1-$e-100-20/result_known.txt 2>&1
        { python3.6 tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e $e -s 100 -n 20 ; } > storage/fnn-1-$e-100-20/result_unknown.txt 2>&1
        cp saved/* storage/fnn-1-$e-100-20/
        printf "Model with embedding size of %i is saved and evaluated.\n" $e
        python3.6 tagger.py --reset -fq
    fi
done

printf "\n## Hidden Layer Size:\n"
for s in {50..400..50}; do
    if [ ! -d "storage/fnn-1-50-$s-20" ]; then
        mkdir storage/fnn-1-50-$s-20
        { time python3.6 tagger.py --train data/training.corpus -p 1 -e 50 -s $s -n 20 ; } > storage/fnn-1-50-$s-20/training.txt 2>&1
        { python3.6 tagger.py --evaluate data/evaluation_known.txt -p 1 -e 50 -s $s -n 20 ; } > storage/fnn-1-50-$s-20/result_known.txt 2>&1
        { python3.6 tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e 50 -s $s -n 20 ; } > storage/fnn-1-50-$s-20/result_unknown.txt 2>&1
        cp saved/* storage/fnn-1-50-$s-20/
        printf "Model with hidden layer size of %i is saved and evaluated.\n" $s
        python3.6 tagger.py --reset -fq
    fi
done

printf "\n## Training Epochs:\n"
for n in {20..200..20}; do
    if [ ! -d "storage/fnn-1-50-100-$n" ]; then
        mkdir storage/fnn-1-50-100-$n
        { time python3.6 tagger.py --train data/training.corpus -p 1 -e 50 -s 100 -n $n ; } > storage/fnn-1-50-100-$n/training.txt 2>&1
        { python3.6 tagger.py --evaluate data/evaluation_known.txt -p 1 -e 50 -s 100 -n $n ; } > storage/fnn-1-50-100-$n/result_known.txt 2>&1
        { python3.6 tagger.py --evaluate data/evaluation_unknown.txt -p 1 -e 50 -s 100 -n $n ; } > storage/fnn-1-50-100-$n/result_unknown.txt 2>&1
        cp saved/* storage/fnn-1-50-100-$n/
        printf "Model trained with %i epochs is saved and evaluated.\n" $n
        python3.6 tagger.py --reset -fq
    fi
done
