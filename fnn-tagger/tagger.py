"""
This tagger module provides the training of a language model via a Feed-forward Neural Network
and the assignment of tags to the words of an untagged sentence.

@see https://www.tensorflow.org/get_started/mnist/pros
@see https://github.com/mrahtz/tensorflow-pos-tagger
"""

import sys
import tensorflow as tf

class Tagger:
	"""A tagger class that trains a Feed-forward Neural Network when instantiated"""

	def __init__(self, training_file_path, vocab_size, n_past_words, embedding_size, n_pos_tags):
		""" Takes in the file path to a training file and returns a Tagger object that is able to train and tag sentences"""

		sys.stdout.write('Initializing model ...\n')

		self.training_file_path = training_file_path

		self.input_x = tf.placeholder(tf.int32, [None, n_past_words + 1], name="input_x")
		self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
		self.embedding_matrix = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))

		# scope: model
		self.word_matrix = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)
		# stack the rows
		# -1: "figure out the right size" (accounts for variable batch size)
		self.feature_vector = tf.reshape(self.word_matrix, [-1, (n_past_words + 1) * embedding_size])

		# one hidden layer
		feature_vector_size = int(self.feature_vector.shape[1])
		h_size = 100
		w1 = tf.Variable(tf.truncated_normal([feature_vector_size, h_size], stddev=0.1))
		self.h = tf.nn.relu(tf.matmul(self.feature_vector, w1))
		self.w2 = tf.Variable(tf.truncated_normal([h_size, n_pos_tags], stddev=0.1))
		self.logits = tf.matmul(self.h, self.w2)
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits))

		# scope accuracy
		# logits has shape [?, n_pos_tags]
		self.predictions = tf.argmax(self.logits, axis=1, name='predictions')
		correct_prediction = tf.equal(self.predictions, self.input_y)
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


	def train(self):
		"""Trains a language model with a given training file"""

		sys.stdout.write('Training starts on "' + self.training_file_path + '" ...\n')

	def tag(self, sentence):
		"""Tags the given sentence"""

		sys.stdout.write(sentence)
		# return self._tagger.tag(helper.tokenize(sentence, True))



# The default tagger
t = Tagger(training_file_path='test/path', vocab_size=50000, n_past_words=3, embedding_size=100, n_pos_tags=20)
t.train()
t.tag('test sentence\n')