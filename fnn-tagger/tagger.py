"""
This tagger module provides the training of a language model via a Feed-forward Neural Network
and the assignment of tags to the words of a given untagged sentence.

@see https://www.tensorflow.org/get_started/mnist/pros
@see https://www.tensorflow.org/versions/master/api_docs/python/tf/nn
@see https://github.com/mrahtz/tensorflow-pos-tagger
"""

import sys
import tensorflow as tf
import numpy as np
import os
import time

import loader
import model

class Tagger:
	"""A tagger class that trains a Feed-forward Neural Network when instantiated"""


	def __init__(self, training_file_path, vocab_size, n_past_words, embedding_size, h_size, test_ratio, batch_size, n_epochs, evaluate_every, checkpoint_every):
		""" Takes in the file path to a training file and returns a Tagger object that is able to train and tag sentences"""

		# set path to the training data, vocabulary size and the number of words to include into training (n_past_words)
		self.training_file_path = training_file_path
		self.vocab_size = vocab_size
		self.n_past_words = n_past_words
		self.embedding_size = embedding_size
		self.h_size = h_size
		self.test_ratio = test_ratio
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.evaluate_every = evaluate_every
		self.checkpoint_every = checkpoint_every

	def train(self):
		"""Trains and evaluates a language model on a given training file"""

		sys.stdout.write('Training starts ...\n')

		# start TensorFlow session
		sess = tf.Session()
		# load necessary training data
		train_batches, test_data, n_pos_tags = self.__load_data()
		x_test = test_data['x']
		y_test = test_data['y']
		# initialize model
		sys.stdout.write('Initializing model ...\n')
		fnn_model, train_op, global_step = self.__model_init(self.vocab_size, self.embedding_size, self.n_past_words, n_pos_tags)
		train_summary_ops, test_summary_ops, summary_writer = self.__logging_init(fnn_model, sess.graph)
		saver = self.__checkpointing_init()

		sess.run(tf.global_variables_initializer())
		sess.graph.finalize()

		standard_ops = [global_step, fnn_model.loss, fnn_model.accuracy]
		train_ops = [train_op, train_summary_ops]
		test_ops = [test_summary_ops]

		for batch in train_batches:
			x_batch, y_batch = zip(*batch)
			self.__step(sess, fnn_model, standard_ops, train_ops, test_ops, x_batch, y_batch, summary_writer, train=True)
			current_step = tf.train.global_step(sess, global_step)

			if current_step % self.evaluate_every == 0:
				print("\nEvaluation:")
				self.__step(sess, fnn_model, standard_ops, train_ops, test_ops, x_test, y_test, summary_writer, train=False)
				print("")

			if current_step % self.checkpoint_every == 0:
				path = saver.save(sess, 'data/model', global_step=current_step)
				print("Saved model checkpoint to '%s'" % path)


	def tag(self, sentence):
		"""Tags the given sentence"""

		sys.stdout.write(sentence)


	def __load_data(self):
		"""Loads and processes training data from a given training file"""

		sys.stdout.write('Loading training data from "' + self.training_file_path + '" ...\n')
		# open training file
		with open(self.training_file_path, 'r') as f:
			tagged_sentences = f.read()

		# set vocabulary and tensor paths
		vocab_path = 'data/vocabulary'
		tensor_path = 'data/tensors'

		textloader = loader.TextLoader(tagged_sentences, self.vocab_size, self.n_past_words, vocab_path, tensor_path)

		x = textloader.features
		y = textloader.labels
		n_pos_tags = len(textloader.pos_to_id)

		idx = int(self.test_ratio * len(x))
		x_test, x_train = x[:idx], x[idx:]
		y_test, y_train = y[:idx], y[idx:]

		train_batches = self.__batch_iterator(list(zip(x_train, y_train)), self.n_epochs)
		test_data = {'x': x_test, 'y': y_test}

		return (train_batches, test_data, n_pos_tags)


	def __batch_iterator(self, data, num_epochs, shuffle=True):
		"""
		Generates a batch iterator for a dataset.
		"""
		data = np.array(data)
		data_size = len(np.atleast_1d(data))
		num_batches_per_epoch = int((data_size-1)/self.batch_size) + 1
		for epoch in range(num_epochs):
			# Shuffle the data at each epoch
			if shuffle:
				shuffle_indices = np.random.permutation(np.arange(data_size))
				shuffled_data = data[shuffle_indices]
			else:
				shuffled_data = data
			for batch_num in range(num_batches_per_epoch):
				start_index = batch_num * self.batch_size
				end_index = min((batch_num + 1) * self.batch_size, data_size)
				yield shuffled_data[start_index:end_index]


	def __model_init(self, vocab_size, embedding_size, n_past_words, n_pos_tags):
		"""
		Initializes a Feed-Forward Neural Network model
		"""
		fnn_model = model.FnnModel(vocab_size, n_past_words, embedding_size, self.h_size, n_pos_tags)
		global_step = tf.Variable(initial_value=0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer()
		train_op = optimizer.minimize(fnn_model.loss, global_step=global_step)

		return fnn_model, train_op, global_step


	def __logging_init(self, fnn_model, graph):
		"""
		Set up logging so that progress can be visualised in TensorBoard.
		"""
		# Add ops to record summaries for loss and accuracy...
		train_loss = tf.summary.scalar("train_loss", fnn_model.loss)
		train_accuracy = tf.summary.scalar("train_accuracy", fnn_model.accuracy)
		# ...then merge these ops into one single op so that they easily be run
		# together
		train_summary_ops = tf.summary.merge([train_loss, train_accuracy])
		# Same ops, but with different names, so that train/test results show up
		# separately in TensorBoard
		test_loss = tf.summary.scalar("test_loss", fnn_model.loss)
		test_accuracy = tf.summary.scalar("test_accuracy", fnn_model.accuracy)
		test_summary_ops = tf.summary.merge([test_loss, test_accuracy])

		timestamp = int(time.time())
		run_log_dir = os.path.join('logs', str(timestamp))
		os.makedirs(run_log_dir)
		# (this step also writes the graph to the events file so that
		# it shows up in TensorBoard)
		summary_writer = tf.summary.FileWriter(run_log_dir, graph)

		return train_summary_ops, test_summary_ops, summary_writer


	def __checkpointing_init(self):
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
		return saver


	def __step(self, sess, model, standard_ops, train_ops, test_ops, x, y, summary_writer, train):
		feed_dict = {model.input_x: x, model.input_y: y}

		if train:
			step, loss, accuracy, _, summaries = sess.run(standard_ops + train_ops, feed_dict)
		else:
			step, loss, accuracy, summaries = sess.run(standard_ops + test_ops, feed_dict)

		print("Step %d: loss %.1f, accuracy %d%%" % (step, loss, 100 * accuracy))
		summary_writer.add_summary(summaries, step)


# The default tagger
t = Tagger(
	training_file_path='data/test.corpus',
	vocab_size=50000,
	n_past_words=3,
	embedding_size=50,
	h_size=100,
	test_ratio=0.1,
	batch_size=64,
	n_epochs=100,
	evaluate_every=100,
	checkpoint_every=100)

# only execute training when file is invoked as a script and not just imported
if __name__ == "__main__":
	t.train()
	t.tag('Module im Master Eletrotechnik\n')