"""
The tagger module provides the training of a language model via a Feed-forward Neural Network
and the assignment of tags to the words of an untagged sentence.
"""
import os
import sys

class Tagger:
    """A tagger class that trains a Feed-forward Neural Network when instantiated"""

    def __init__(self, training_file_path):
        """ Takes in the file path to a training file and returns a Tagger object
        that is able to tag sentences"""


        # if settings.LOAD_HMM:
        #     print("Loading HMM from file...")
        #     try:
        #         with open(settings.HMM_LOCATION, "rb") as tagger_input:
        #             self._tagger = dill.load(tagger_input)
        #     except:
        #         print("Could not load HMM from file! Now creating one...")

        # if not self._tagger:
        #     training_file = ""

        #     # Open the training file
        #     with open(training_file_path, encoding="utf-8") as f:
        #         training_file = f.readlines()


        #     print("Generating the training set from file '%s'..." % training_file_path.split("/")[-1])
        #     training_set = loader.generate_training_set(training_file)

        #     temp_list = [list(zip(words, tags)) for words, tags in training_set]

        #     if settings.HMM_ADVANCED_TRAINING:
        #         print("Evaluating each training sentence!\nThis may take several hours!")
        #         new_list = []
        #         temp_list_length = len(temp_list)

        #         for i, entry in enumerate(temp_list):

        #             # Debug output
        #             if i%100 == 0:
        #                 percentage = (i/temp_list_length)*100
        #                 sys.stdout.write('\r')
        #                 sys.stdout.write("[%-100s] %.2f%%" % ('='*int(percentage), percentage))
        #                 sys.stdout.flush()

        #             # Check for result
        #             if gives_query_result(entry):
        #                 new_list.append(entry)

        #         print("Old size / new size: %d/%d" % (len(temp_list), len(new_list)))
        #         temp_list = new_list

        #     self._tagger = TAGGING_CLASS.train(temp_list)

        #     if settings.SAVE_HMM:
        #         print("Saving the HMM...")
        #         with open(settings.HMM_LOCATION, "wb") as output:
        #             dill.dump(self._tagger, output)

    def train(training_file_path):
        """Trains a language model with a given training file"""

    def tag(self, sentence):
        """Tags the given sentence"""

        # return self._tagger.tag(helper.tokenize(sentence, True))





# The default tagger
t = Tagger(settings.TRAINING_DATA_PATH)

if __name__ == '__main__':
    path = os.path.dirname(os.path.realpath(__file__))
    t = Tagger(path + "/data/training_data.txt")
