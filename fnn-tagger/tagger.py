"""
This tagger module provides the training of a language model via a Feed-forward Neural Network
and the assignment of tags to the words of a given untagged sentence.

@see https://www.tensorflow.org/get_started/mnist/pros
@see https://www.tensorflow.org/versions/master/api_docs/python/tf/nn

The code is based on the TensorFlow Part-of-Speech Tagger from Matthew Rahtz
@see https://github.com/mrahtz/tensorflow-pos-tagger
"""

import tensorflow as tf
import numpy as np
import os
import time

import loader
import model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Tagger:
	"""
	A Part-of-Speech tagging class that trains a Feed-forward Neural Network with already tagged data
	and applies POS tags to untagged sentences
	"""


	def __init__(self, training_file_path, vocab_size, n_past_words, embedding_size, h_size, test_ratio, batch_size, n_epochs, evaluate_every, checkpoint_every):
		"""
		Takes in the file path to a training file and returns a Tagger object that is able to train and tag sentences

		@param training_file_path: Path to a file with tagged sentences of this form: word1|TAG word2|TAG ...
		@param vocab_size: Dimension of the vocabulary (number of distinct words)
		@param n_past_words: Number of preceding words to take into account for the POS tag training of the current word
		@param embedding_size: Dimension of the word embeddings
		@param h_size: Dimension of the hidden layer
		@param test_ratio: Ratio of test data extracted from the training data
		@param batch_size: Size of the training batches
		@param n_epochs: Number of training epochs
		@param evaluate_every: Show evaluation result after this number of trainings steps
		@param checkpoint_every: Save model state after this number of trainings steps
		"""
		# initialize given parameters
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

		# set vocabulary and tensor files for saving and loading
		self.vocab_path = 'saved/vocabulary'
		self.tensor_path = 'saved/tensors'


	def train(self):
		"""
		Trains and evaluates a language model on a given training file
		"""
		# start tensorflow session
		print('Training starts ...')
		sess = tf.Session()

		# get training and test data and the number of existing POS tags
		train_batches, test_data, n_pos_tags = self.__load_data()
		x_test = test_data['x']
		y_test = test_data['y']

		print('Initializing model ...')
		# initialize the model by specifying initial values for the tensorflow variables
		fnn_model, train_op, global_step = self.__model_init(self.vocab_size, self.embedding_size, self.n_past_words, n_pos_tags)
		train_summary_ops, test_summary_ops, summary_writer = self.__logging_init(fnn_model, sess.graph)
		saver = self.__checkpointing_init()

		# take the initial values that have already been specified, and assign them to each Variable
		sess.run(tf.global_variables_initializer())
		# make the graph read-only to ensure that no operations are added to it when shared between multiple threads
		sess.graph.finalize()

		standard_ops = [global_step, fnn_model.loss, fnn_model.accuracy]
		train_ops = [train_op, train_summary_ops]
		test_ops = [test_summary_ops]

		# start training with taking one training batch each step
		for batch in train_batches:
			x_batch, y_batch = zip(*batch)
			self.__step(sess, fnn_model, standard_ops, train_ops, test_ops, x_batch, y_batch, summary_writer, train=True)
			current_step = tf.train.global_step(sess, global_step)

			if current_step % self.evaluate_every == 0:
				self.__step(sess, fnn_model, standard_ops, train_ops, test_ops, x_test, y_test, summary_writer, train=False)

			if current_step % self.checkpoint_every == 0:
				path = saver.save(sess, 'saved/model', global_step=current_step)
				print("Saved model checkpoint to '%s'" % path)


	def tag(self, sentence):
		"""
		Tags a given sentence with the help of a previously trained model

		@param sentence: a string of space separated words, like "word1 word2 wore3"
		@return a string of space separated word-tag tuples, like "word1|TAG word2|TAG wird3|TAG"
		"""
		data = loader.TextLoader(sentence, self.vocab_size, self.n_past_words, self.vocab_path)

		# start tensorflow session
		sess = tf.Session()

		# load saved session
		checkpoint_file = tf.train.latest_checkpoint('saved/')
		saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
		saver.restore(sess, checkpoint_file)

		# load input words and predictions from the saved session
		graph = tf.get_default_graph()
		input_x = graph.get_operation_by_name("input_x").outputs[0]
		predictions = graph.get_operation_by_name("predictions").outputs[0]
		predicted_pos_ids = sess.run(predictions, feed_dict={input_x: data.features})

		# create lists of the sentence words and their corresponding predicted POS tags
		words = []
		for sentence_word_ids in data.features:
			word_id = sentence_word_ids[0]
			words.append(data.id_to_word[word_id])
		predicted_pos = []
		for pred_id in predicted_pos_ids:
			predicted_pos.append(data.id_to_pos[pred_id])

		# merge word and tag lists
		word_pos_tuples = zip(words, predicted_pos)
		annotated_words = []
		for tup in word_pos_tuples:
			annotated_word = '%s/%s' % (tup[0], tup[1])
			annotated_words.append(annotated_word)

		return ' '.join(annotated_words)


	def __load_data(self):
		"""
		Loads and processes training data from a given training file

		@return train_batches: training data splitted into iterable batches
		@return test_data: data to test the trained model
		@return n_pos_tags: total number of existing POS tags
		"""
		# read tagged data from training file
		print('Loading training data from "' + self.training_file_path + '" ...')
		with open(self.training_file_path, 'r') as f:
			tagged_sentences = f.read()
			f.close()

		data = loader.TextLoader(tagged_sentences, self.vocab_size, self.n_past_words, self.vocab_path, self.tensor_path)
		x = data.features
		y = data.labels
		n_pos_tags = len(data.pos_to_id)

		# split data for training and testing according to the test ratio
		idx = int(self.test_ratio * len(x))
		x_test, x_train = x[:idx], x[idx:]
		y_test, y_train = y[:idx], y[idx:]

		# create iterable training batches
		train_batches = self.__batch_iterator(list(zip(x_train, y_train)), self.n_epochs)
		test_data = {'x': x_test, 'y': y_test}

		return (train_batches, test_data, n_pos_tags)


	def __batch_iterator(self, data, num_epochs, shuffle=True):
		"""
		Generates a batch iterator for a dataset
		"""
		data = np.array(data)
		data_size = len(np.atleast_1d(data))
		num_batches_per_epoch = int((data_size-1)/self.batch_size) + 1
		for epoch in range(num_epochs):
			# shuffle the data at each epoch
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
		Set up logging that the progress can be visualised in TensorBoard
		"""
		# Add ops to record summaries for loss and accuracy...
		train_loss = tf.summary.scalar("train_loss", fnn_model.loss)
		train_accuracy = tf.summary.scalar("train_accuracy", fnn_model.accuracy)
		# ...then merge these ops into one single op so that they easily be run together
		train_summary_ops = tf.summary.merge([train_loss, train_accuracy])
		# Same ops, but with different names, so that train/test results show up separately in TensorBoard
		test_loss = tf.summary.scalar("test_loss", fnn_model.loss)
		test_accuracy = tf.summary.scalar("test_accuracy", fnn_model.accuracy)
		test_summary_ops = tf.summary.merge([test_loss, test_accuracy])

		timestamp = int(time.time())
		run_log_dir = os.path.join('logs', str(timestamp))
		os.makedirs(run_log_dir)
		# (this step also writes the graph to the events file so that it shows up in TensorBoard)
		summary_writer = tf.summary.FileWriter(run_log_dir, graph)

		return train_summary_ops, test_summary_ops, summary_writer


	def __checkpointing_init(self):
		"""
		Initialize the Saver class to save and restore variables
		"""
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
		return saver


	def __step(self, sess, model, standard_ops, train_ops, test_ops, x, y, summary_writer, train):
		"""
		Execute one training step
		"""
		feed_dict = {model.input_x: x, model.input_y: y}

		if train:
			step, loss, accuracy, _, summaries = sess.run(standard_ops + train_ops, feed_dict)
		else:
			print('')
			step, loss, accuracy, summaries = sess.run(standard_ops + test_ops, feed_dict)

		print("Step %d: loss %.1f, accuracy %d%%" % (step, loss, 100 * accuracy), end="\r", flush=True)
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
	# print(t.tag('Module im Bachelor Informatik'))