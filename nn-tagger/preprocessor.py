"""
The preprocessor module provides preprocessing and preparation to train a language model
"""

import os, random, argparse
import pickle

class Corpus:
    """
    This class contains all methods for corpus preprocessing and preparation for training
    """

    def create_from_training_set(self, training_file, target_file):
        """
        creates a corpus file from an already created hmm training set 
        The new corpus contains one tagged sentence per line of format: word1/TAG word2/TAG ...

        @param training_file:  Path to a training_set file from hmm
        @param target_file:    File to store the corpus in
        """

        # check if training set file exists
        if not os.path.isfile(training_file):
            print('Error: corpus file "%s" doesn\'t exist' % training_file)
            return

        print('Loading training file...')
        with open(training_file,'rb') as training_data:
            training_set = pickle.load(training_data)
            training_data.close()
        
        print('Processing training data...')
        tagged_lines = []
        for words, tags in training_set:
            tagged_lines.append(' '.join([i+'/'+j for i,j in zip(words, tags)]))
        
        print('Saving corpus file...')
        with open(target_file,'w') as target:
            for line in tagged_lines:
                target.write("%s\n" % line)
        print('Done.')


    def line_processing(self, corpus_file_path, delete_lines_containing='', randomize=False):
        """
        operations for the lines of an existing corpus like special line deletion and randomisation

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
            print('Shuffling lines...')
            lines = open(corpus_file_path).readlines()
            random.shuffle(lines)
            open(corpus_file_path, 'w').writelines(lines)
            print('Done.')
            

    def parse_args(self):
        """
        Get script arguments
        """

        parser = argparse.ArgumentParser()
        parser.add_argument("--source", type=str, help="Path to a training_set file from hmm")
        parser.add_argument("--target", type=str, help="File to store the created corpus in")
        parser.add_argument("-s", "--shuffle", type=str, help="Shuffles all lines in given corpus file")

        return parser.parse_args()


# only execute corpus processing when file is invoked as a script and not just imported
if __name__ == "__main__":
    # create corpus instance
    c = Corpus()
    
    args = c.parse_args()
    # invoke corpus creation
    if args.source is not None and args.target is not None:
        c.create_from_training_set(args.source, args.target)
    # shuffle corpus
    if args.shuffle is not None:
        c.line_processing(args.shuffle, randomize=True)
