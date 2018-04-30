"""
The vocabulary module provides information about the vocabulary of a language model
"""

import os, random, argparse
from texttable import Texttable

class CorpusVocabulary:
    """
    This class contains all methods for corpus vocabulary retrieval and information
    """


    def inform(self, corpus_file_path):
        """
        Processes corpus and prints information

        @param corpus_file_path: Path to a file with tagged sentences of this form: word1/TAG word2/TAG ...
        """

        # check if corpus file exists
        if not os.path.isfile(corpus_file_path):
            print('Error: corpus file "%s" doesn\'t exist' % corpus_file_path)
            return
        
        # get corpus vocabulary information
        with open(corpus_file_path, 'r') as f:
            lines = f.readlines()
            line_count = len(lines)
            word_count = 0
            words, tags = {}, {}
            for line in lines:
                word_count += len(line.split())
                for t in line.split():
                    if t.find('/') > -1:
                        tup = t.rsplit('/',1)
                        word, tag = tup[0], tup[1]
                        words[word] = words[word] + 1 if word in words and words[word] > 0 else 1
                        tags[tag] = tags[tag] + 1 if tag in tags and tags[tag] > 0 else 1
            f.close()
        
        # print counting results
        print('# COUNTS:\n')
        table = Texttable(120)
        table.set_deco(Texttable.HEADER)
        table.set_cols_dtype(['i', 't'])
        table.set_cols_align(['r', 'l'])
        table.add_row([line_count, 'sentences'])
        table.add_row([word_count, 'words'])
        table.add_row([len(words), 'distinct words'])
        table.add_row([len(tags), 'distinct tags'])
        print(table.draw())
            
        # print tag counts
        print('\n# TAGS:\n')
        table = Texttable(120)
        table.set_deco(Texttable.HEADER)
        table.set_cols_dtype(['i', 't'])
        table.set_cols_align(['r', 'l'])
        for tag, count in sorted(tags.items(), key=lambda x: x[1], reverse=True):
            table.add_row([count, tag])
        print(table.draw())


    def parse_args(self):
        """
        Get script arguments
        """

        parser = argparse.ArgumentParser()
        parser.add_argument("--corpus", type=str, help="Corpus file to process and inform about")

        return parser.parse_args()


# only execute training when file is invoked as a script and not just imported
if __name__ == "__main__":
    # create tagger instance
    v = CorpusVocabulary()
    
    args = v.parse_args()
    # invoke training
    if args.corpus is not None:
        v.inform(args.corpus)