import dill
import hmmtagger
import nltk
from texttable import Texttable
from sklearn.metrics import cohen_kappa_score

# initialize counter variables
n_sentences_correct = n_words_correct = n_tags_correct = 0
tags_wrong = {}
true_tags, predicted_tags = [], []
# get pre-tagged evaluation data
text = open('evaluation_known.txt').read()
text = '\n'.join([s for s in text.splitlines() if s and s[0] != '#'])
tagged_sentences = text.splitlines()
n_sentences = len(tagged_sentences)
# tag data based on trained language model
hmm_file = open('hmmtagger/data/hmm', 'rb')
tagger = dill.load(hmm_file)
# tagger.tag(['alle', 'module', 'vom', 'master', 'informatik'])
sentences = []
for sentence in text.splitlines():
    words = []
    for word in sentence.split():
        words.append(word[:(word.index('/'))])
    sentences.append(words)
computed_words_list = []
for sentence in sentences:
    computed_words_list.append(tagger.tag(sentence))
computed_words = []
for s in computed_words_list:
    for a,b in s:
        computed_words.append(a + '/' + b)
computed_sentences = []
sentence_lengths = [len(x.split()) for x in tagged_sentences]
n_words = sum(sentence_lengths)
for i, l in enumerate(sentence_lengths):
    computed_sentences.append(' '.join(computed_words[sum(sentence_lengths[:i]):sum(sentence_lengths[:i])+l]))

# start evaluation
print('Evaluation starts...')

pos = []
# compute correct sentences and words
for t, c in zip(tagged_sentences, computed_sentences):
    t, c = t.strip(), c.strip()
    # check sentences
    if t == c:
        n_sentences_correct += 1
        n_words_correct += len(t.split())
        n_tags_correct += len(t.split())
    else:
        # check words and tags
        for tw, cw in zip (t.split(), c.split()):
            tag1 = tw[tw.index('/')+1:]
            tag2 = cw[cw.index('/')+1:]
            if tag1 not in pos:
                pos.append(tag1)
            if tag2 not in pos:
                pos.append(tag2)
            if tw != cw:
                # build a custom key: correct_word/tag|wrong_word/tag to count its occurence
                key = tw + '|' + cw
                tags_wrong[key] = tags_wrong[key] + 1 if key in tags_wrong and tags_wrong[key] > 0 else 1
            # increment counter for matching words and tags
            n_words_correct += 1 if tw[:tw.index('/')] == cw[:cw.index('/')] else 0
            n_tags_correct += 1 if tag1 == tag2 else 0
            # add tag ids to true and predicted tag lists for cohens kappa
            true_tags.append(pos.index(tag1)) 
            predicted_tags.append(pos.index(tag2))
# calculate cohen's kappa
kappa = cohen_kappa_score(true_tags, predicted_tags)
# print ratio of correct sentences, words and tags
print('\n# RESULTS:\n')
table = Texttable(120)
table.set_deco(Texttable.HEADER)
table.set_cols_dtype(['t', 'f', 't'])
table.set_cols_align(['r', 'c', 'l'])
table.add_row([str(n_sentences_correct) + ' / ' + str(n_sentences), n_sentences_correct/n_sentences, 'sentences correct'])
table.add_row([str(n_words_correct) + ' / ' + str(n_words), n_words_correct/n_words, 'words recognized'])
table.add_row([str(n_tags_correct) + ' / ' + str(n_words), n_tags_correct/n_words, 'tags correct'])
table.add_row(['', kappa, 'kappa score'])
print(table.draw())
# show wrong tags
print('\n# ERRORS:\n')
table = Texttable(120)
table.set_deco(Texttable.HEADER)
table.set_cols_dtype(['i', 't', 't'])
table.set_cols_align(['c', 'l', 'l'])
table.set_chars(['-', '|', '+', '-'])
table.add_rows([["count", "expected", "computed"]])
for key, count in sorted(tags_wrong.items(), key=lambda x: x[1], reverse=True):
    table.add_row([count, key[:key.index('|')], key[key.index('|')+1:]])
print(table.draw())
