"""
This tagger module provides the training of a language model via a Feed-forward Neural Network
and the assignment of tags to the words of a given untagged sentence.

@see https://www.tensorflow.org/get_started/mnist/pros
@see https://www.tensorflow.org/versions/master/api_docs/python/tf/nn
@see https://github.com/mrahtz/tensorflow-pos-tagger
"""

import sys
import tensorflow as tf

import model

class Tagger:
	"""A tagger class that trains a Feed-forward Neural Network when instantiated"""

	def __init__(self, training_file_path, vocab_size, n_past_words, embedding_size, h_size):
		""" Takes in the file path to a training file and returns a Tagger object that is able to train and tag sentences"""

		# set path to the training data
		self.training_file_path = training_file_path

		# TODO
		n_pos_tags = 20

		# initialize input word vectors. None: "variable size"
		sys.stdout.write('Initializing model ...\n')
		fnn = model.FnnModel(vocab_size, n_past_words, embedding_size, h_size, n_pos_tags)


	def train(self):
		"""Trains a language model with a given training file"""

		sys.stdout.write('Training starts on "' + self.training_file_path + '" ...\n')

	def tag(self, sentence):
		"""Tags the given sentence"""

		sys.stdout.write(sentence)
		# return self._tagger.tag(helper.tokenize(sentence, True))



# The default tagger
t = Tagger(training_file_path='test/path', vocab_size=50000, n_past_words=3, embedding_size=50, h_size=100)

# only execute when script is directly called
if __name__ == "__main__":
	t.train()
	t.tag('test sentence\n')