"""
The preprocessor module provides preprocessing and preparation to train a language model
"""

import os, random

class CorpusPreprocessor:
    """
    This class contains all methods for corpus preprocessing and preparation for training
    """


    def __init__(self, corpus_file_path, delete_lines_containing='', randomize=False):
        """
        ...

        @param corpus_file_path:        Path to a file with tagged sentences of this form: word1/TAG word2/TAG ...
        @param delete_lines_containing: All lines containing the given string should be deleted.
        @param randomize:               Shuffle lines of the corpus.
        """

        # check if corpus file exists
        if not os.path.isfile(corpus_file_path):
            print('Error: corpus file "%s" doesn\'t exist' % corpus_file_path)
            return
        
        # custom line deleting
        if delete_lines_containing != '':
            print('Loading corpus data from "%s"...' % corpus_file_path)
            output = []
            line_count_before = 0
            # only keep all lines not containing the needle
            with open(corpus_file_path, 'r') as f:
                lines = f.readlines()
                line_count_before = len(lines)
                for line in lines:
                    if delete_lines_containing not in line:
                        output.append(line)
                f.close()
            # write lines to corpus file
            with open(corpus_file_path, 'w') as f:
                f.writelines(output)
                f.close()
            print('Deleted %i of %i lines (%.1f%%)' % (line_count_before-len(output), line_count_before, (line_count_before-len(output))/line_count_before*100))

        # randomize lines
        if randomize:
            lines = open(corpus_file_path).readlines()
            random.shuffle(lines)
            open(corpus_file_path, 'w').writelines(lines)
            

CorpusPreprocessor('fnn-tagger/data/hmm.random.corpus', 'welche/R_LIST module/M_MTSModule werden/X von/X prof/X_Person', True)