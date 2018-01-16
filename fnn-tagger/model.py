"""
This module provides the structure and initialization of a Feed-forward Neural Network model

@see https://www.tensorflow.org/get_started/mnist/pros
@see https://www.tensorflow.org/versions/master/api_docs/python/tf/nn
@see https://github.com/mrahtz/tensorflow-pos-tagger
"""

import tensorflow as tf

class FnnModel:
	"""
    A tagger class that trains a Feed-forward Neural Network when instantiated
    """

	def __init__(self, vocab_size, n_past_words, embedding_size, h_size, n_pos_tags):
		"""
        Initializes the Feed-forward Neural Network model
        """

		# initialize input word vectors. None: "variable size"
		self.input_x = tf.placeholder(tf.int32, [None, n_past_words + 1], name="input_x")
		# initialize input lables (tags)
		self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
		# initialize an embedding matrix with random truncated normal values
		self.embedding_matrix = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))

		# build word matrix out of a lookup in the embedding matrix for the input words
		self.word_matrix = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)
		# stack the rows to create one vector. -1: "figure out the right size" (accounts for variable batch size)
		self.feature_vector = tf.reshape(self.word_matrix, [-1, (n_past_words + 1) * embedding_size])

		# get size of the feature vector
		feature_vector_size = int(self.feature_vector.shape[1])
		# initialize weights between input layer and hidden layer
		w1 = tf.Variable(tf.truncated_normal([feature_vector_size, h_size], stddev=0.1))
		# compute rectified linear activation function on hidden layer
		self.h = tf.nn.relu(tf.matmul(self.feature_vector, w1))
		# initialize weights between hidden layer and output layer
		self.w2 = tf.Variable(tf.truncated_normal([h_size, n_pos_tags], stddev=0.1))
		# compute the logits for the output layer with shape [?, n_pos_tags]
		self.logits = tf.matmul(self.h, self.w2)
		# compute the loss
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits))

		# reduce the logits output with the largest value to the output layer size = predictions 
		self.predictions = tf.argmax(self.logits, axis=1, name='predictions')
		# get the correct predictions by comparing to the given labels
		correct_prediction = tf.equal(self.predictions, self.input_y)
		# compute the overall accuracy
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))