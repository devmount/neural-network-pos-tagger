"""
This module provides a class to load text data for training,
to generate a vocabulary and save this vocabulary and generated tensors.

@see https://www.tensorflow.org/get_started/mnist/pros
@see https://www.tensorflow.org/versions/master/api_docs/python/tf/nn

The code is based on the TensorFlow Part-of-Speech Tagger from Matthew Rahtz
@see https://github.com/mrahtz/tensorflow-pos-tagger/blob/master/data_utils.py
"""

import os
from collections import Counter
import pickle

UNKNOWN_WORD_ID = 0
UNKNOWN_WORD = "<UNKNOWN_WORD>"

UNTAGGED_POS_ID = 0
UNTAGGED_POS = "<UNTAGGED_POS>"

class TextLoader:
    """
    A class that loads annotated training data, generates a vocabulary of words and of POS tags
    and provides loading and saving of this vocabulary and generated tensors
    """


    def __init__(self, sentences, vocab_size, n_past_words, vocab_path, tensor_path=None, silent=False):
        self.vocab_size = vocab_size
        self.n_past_words = n_past_words

        if os.path.exists(vocab_path):
            if not silent: print("Loading saved vocabulary...")
            self.load_vocab(vocab_path)
        else:
            if not silent: print("Generating vocabulary...")
            self.gen_vocab(sentences)
            self.save_vocab(vocab_path)

        if tensor_path is not None and os.path.exists(tensor_path):
            if not silent: print("Loading saved tensors...")
            self.load_tensors(tensor_path)
        else:
            if not silent: print("Generating tensors...")
            self.gen_features_and_labels(sentences)
            if tensor_path is not None:
                self.save_tensors(tensor_path)


    def gen_vocab(self, tagged_sentences):
        words, pos_tags = self.split_sentence(tagged_sentences, drop_untagged=True)

        word_counts = Counter(words)
        unique_pos_tags = set(pos_tags)

        # most_common() returns (word, count) tuples
        # Why the '- 1'? To account for the extra word we add for words not in the vocabulary, UNKNOWN_WORD.
        words_to_keep = [t[0] for t in word_counts.most_common(self.vocab_size - 1)]

        self.word_to_id = {word: i for i, word in enumerate(words_to_keep, start=1)}
        # words not in the vocabulary will be mapped to this word
        self.word_to_id[UNKNOWN_WORD] = UNKNOWN_WORD_ID # = 0

        self.pos_to_id = {pos: i for i, pos in enumerate(list(unique_pos_tags), start=1)}
        self.pos_to_id[UNTAGGED_POS] = UNTAGGED_POS_ID # = 0

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.id_to_pos = {v: k for k, v in self.pos_to_id.items()}

        self.words = words


    def save_vocab(self, vocab_filename):
        dicts = [self.word_to_id, self.pos_to_id, self.id_to_word, self.id_to_pos]
        with open(vocab_filename, 'wb', encoding="utf8") as f:
            pickle.dump(dicts, f)
            f.close()


    def load_vocab(self, vocab_path):
        with open(vocab_path, 'rb', encoding="utf8") as f:
            dicts = pickle.load(f)
            f.close
        self.word_to_id = dicts[0]
        self.pos_to_id = dicts[1]
        self.id_to_word = dicts[2]
        self.id_to_pos = dicts[3]


    def save_tensors(self, tensors_path):
        tensors = [self.features, self.labels]
        with open(tensors_path, 'wb', encoding="utf8") as f:
            pickle.dump(tensors, f)
            f.close()


    def load_tensors(self, tensors_path):
        with open(tensors_path, 'rb', encoding="utf8") as f:
            tensors = pickle.load(f)
            f.close()
        self.features = tensors[0]
        self.labels = tensors[1]


    def gen_features_and_labels(self, tagged_sentences):
        x = []
        y = []

        for sentence in tagged_sentences.split('\n'):
            # Why drop_untagged=False here?
            # Because we might have received an untagged sentence which we now want to tag.
            words, pos_tags = self.split_sentence(sentence, drop_untagged=False)

            for j in range(len(words)):
                if len(pos_tags) != 0:
                    tag = pos_tags[j]
                    y.append(self.pos_to_id[tag])

                past_word_ids = []
                for k in range(0, self.n_past_words+1):
                    if j-k < 0: # out of bounds
                        past_word_ids.append(UNKNOWN_WORD_ID)
                    elif words[j-k] in self.word_to_id:
                        past_word_ids.append(self.word_to_id[words[j-k]])
                    else: # word not in vocabulary
                        past_word_ids.append(UNKNOWN_WORD_ID)
                x.append(past_word_ids)

        self.features = x
        self.labels = y


    def split_sentence(self, tagged_sentence, drop_untagged):
        tagged_words = tagged_sentence.split()
        word_tag_tuples = [x.split("/") for x in tagged_words]

        words = []
        pos_tags = []
        for word_tag_tuple in word_tag_tuples:
            if len(word_tag_tuple) > 2:
                # We've got something like TAG1/TAG2/TAG3
                continue

            if drop_untagged and len(word_tag_tuple) == 1:
                continue

            word = word_tag_tuple[0]
            words.append(word)

            if len(word_tag_tuple) == 1:
                pos_tags.append(UNTAGGED_POS)
            else:
                tag = word_tag_tuple[1]
                pos_tags.append(tag)

        return words, pos_tags
