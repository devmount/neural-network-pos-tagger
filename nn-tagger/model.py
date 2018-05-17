"""
This module provides the structure and initialization of different Neural Network models

@see https://www.tensorflow.org/get_started/mnist/pros
@see https://www.tensorflow.org/versions/master/api_docs/python/tf/nn

The code is based on the TensorFlow Part-of-Speech Tagger from Matthew Rahtz
@see https://github.com/mrahtz/tensorflow-pos-tagger/blob/master/model.py
"""

import tensorflow as tf


class FNN:
    """
    A class that initializes the structure of a Feed-forward Neural Network when instantiated
    """


    def __init__(self, vocab_size, n_past_words, embedding_size, h_size, n_pos_tags):
        """
        Initializes the Feed-forward Neural Network model

        @param vocab_size: Dimension of the vocabulary (number of distinct words)
        @param n_past_words: Number of preceding words to take into account for the POS tag training of the current word
        @param embedding_size: Dimension of the word embeddings
        @param h_size: Dimension of the hidden layer
        @param n_pos_tags: Number of existing POS tags
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

        # initialize weights between input layer and hidden layer
        w1 = tf.Variable(tf.truncated_normal([(n_past_words + 1) * embedding_size, h_size], stddev=0.1))
        # compute rectified linear activation function on hidden layer
        self.h = tf.nn.relu(tf.matmul(self.feature_vector, w1))
        # initialize weights between hidden layer and output layer
        self.w2 = tf.Variable(tf.truncated_normal([h_size, n_pos_tags], stddev=0.1))
        # compute the logits for the output layer with shape [?, n_pos_tags]
        self.logits = tf.matmul(self.h, self.w2)
        # compute the loss
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits))

        # reduce the logits output with the largest value to the output layer size = predictions 
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        # get the correct predictions by comparing to the given labels
        correct_prediction = tf.equal(self.predictions, self.input_y)
        # compute the overall accuracy
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # apply an optimizer
        self.optimizer = tf.train.AdamOptimizer()


class RNN:
    """
    A class that initializes the structure of a Recurrent Neural Network when instantiated
    """


    def __init__(self, vocab_size, n_timesteps, embedding_size, h_size, n_pos_tags):
        """
        Initializes the Recurrent Neural Network model

        @param vocab_size: Dimension of the vocabulary (number of distinct words)
        @param n_timesteps: Number of previous training steps to include
        @param embedding_size: Dimension of the word embeddings
        @param h_size: Dimension of the hidden layer
        @param n_pos_tags: Number of existing POS tags
        """

        # initialize input word vectors of shape [batch_size, n_timesteps, word_id]
        self.input_x = tf.placeholder(tf.int32, [None, n_timesteps], name="input_x")
        # initialize input labels of shape [batch_size, n_timesteps]
        self.input_y = tf.placeholder(tf.int64, [None], name="input_y")

        # initialize an embedding matrix with random truncated normal values
        self.embedding_matrix = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
        # build word matrix out of a lookup in the embedding matrix for the input words
        self.word_matrix = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)
        # stack the rows to create one vector. -1: "figure out the right size" (accounts for variable batch size)
        self.feature_vector = tf.reshape(self.word_matrix, [-1, n_timesteps, embedding_size])

        # initialize hidden layer with shape [h_size, n_pos_tags]
        self.w = tf.Variable(tf.truncated_normal([h_size, n_pos_tags], stddev=0.1))
        # create lstm cell
        cell = tf.nn.rnn_cell.LSTMCell(h_size, activation=tf.nn.relu)
        # calculate outputs with shape [batch_size*n_timesteps, h_size]
        outputs, states = tf.nn.dynamic_rnn(cell, self.feature_vector, dtype=tf.float32)
        # compute the logits for the output layer
        self.logits = tf.matmul(tf.reshape(outputs, [-1, h_size]), self.w)
        # reshape logits to make the first dimension fit the batch size/labels: [batch_size, n_timesteps*n_pos_tags]
        logits = tf.reshape(self.logits, [-1, n_timesteps*n_pos_tags])
        # compute the loss
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))

        # reduce the logits output with the largest value to the output layer size = predictions 
        self.predictions = tf.argmax(logits, axis=1, name="predictions")
        # get the correct predictions by comparing to the given labels
        correct_prediction = tf.equal(self.predictions, self.input_y)
        # compute the overall accuracy
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # apply an optimizer
        self.optimizer = tf.train.AdamOptimizer()