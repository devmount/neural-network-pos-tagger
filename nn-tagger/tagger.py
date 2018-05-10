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
import os, shutil, time, argparse
from texttable import Texttable
from sklearn.metrics import cohen_kappa_score

import settings as conf
from loader import TextLoader
import model

class Tagger:
    """
    A Part-of-Speech tagging class that trains a Feed-forward Neural Network with already tagged data
    and applies POS tags to untagged sentences
    """


    def __init__(self, architecture=conf.ARCHITECTURE, n_past_words=conf.N_PAST_WORDS, embedding_size=conf.EMBEDDING_SIZE, h_size=conf.HIDDEN_LAYER_SIZE, n_epochs=conf.N_EPOCHS, n_timesteps=conf.N_TIMESTEPS):
        """
        Takes in the file path to a training file and returns a Tagger object that is able to train and tag sentences
        """

        # initialize given parameters
        self.architecture = architecture
        self.vocab_size = conf.VOCAB_SIZE
        self.n_past_words = n_past_words if architecture == 'FNN' else 0
        self.n_timesteps = n_timesteps
        self.embedding_size = embedding_size
        self.h_size = h_size
        self.test_ratio = conf.TEST_RATIO
        self.batch_size = conf.BATCH_SIZE
        self.n_epochs = n_epochs
        self.checkpoint_every = conf.CHECKPOINT_EVERY

        # set vocabulary and tensor files for saving and loading
        self.log_dir = 'logs'
        self.storage_dir = 'saved'
        self.vocab_path = os.path.join(self.storage_dir, 'vocabulary')
        self.tensor_path = os.path.join(self.storage_dir, 'tensors')

        self.data = None


    def train(self, training_file_path):
        """
        Trains and evaluates a language model on a given training file

        @param training_file_path: Path to a corpus file with tagged sentences of format: word1/TAG word2/TAG with one sentence per line
        """

        # check if training file exists
        if not os.path.isfile(training_file_path):
            print('Error: training file "%s" doesn\'t exist' % training_file_path)
            return

        # get training and test data and the number of existing POS tags
        train_batches, test_data, n_pos_tags = self.__load_data(training_file_path)
        x_test = self.__rnn_reshape(test_data['x']) if self.architecture == 'RNN' else test_data['x']
        y_test = test_data['y']

        # initialize the model by starting session and specifying initial values for the tensorflow variables
        print('Initializing model...')

        # start tensorflow session
        print('Training starts (can be finished earlier with CTRL+C)...')
        sess = tf.Session()

        nn_model, train_op, global_step = self.__model_init(n_pos_tags)
        train_summary_ops, test_summary_ops, summary_writer = self.__logging_init(nn_model, sess.graph)
        saver = self.__checkpointing_init()

        # take the initial values that have already been specified, and assign them to each Variable
        sess.run(tf.global_variables_initializer())
        # make the graph read-only to ensure that no operations are added to it when shared between multiple threads
        sess.graph.finalize()

        standard_ops = [global_step, nn_model.loss, nn_model.accuracy]
        train_ops = [train_op, train_summary_ops]
        test_ops = [test_summary_ops]

        # start training with taking one training batch each step
        for batch in train_batches:
            x_batch, y_batch = zip(*batch)
            # reshape data for RNN time steps
            if self.architecture == 'RNN':
                x_batch = self.__rnn_reshape(x_batch)
            self.__step(sess, nn_model, standard_ops, train_ops, test_ops, x_batch, y_batch, summary_writer, train=True)
            current_step = tf.train.global_step(sess, global_step)

            # checkpoint: evaluate with no training data
            if current_step % self.checkpoint_every == 0:
                self.__step(sess, nn_model, standard_ops, train_ops, test_ops, x_test, y_test, summary_writer, train=False)
                path = saver.save(sess, os.path.join(self.storage_dir, 'model'), global_step=current_step)
                print(" - saved model checkpoint to '%s'" % path)


    def tag(self, sentence, pretty_print=False, silent=False):
        """
        Tags a given sentence with the help of a previously trained model

        @param sentence: a string of space separated words, like "word1 word2 word3"
        @param pretty_print: print tagged sentence
        @param silent: no print output messages
        @return a string of space separated word-tag tuples, like "word1/TAG word2/TAG word3/TAG"
        """

        self.data = TextLoader(self.__replace(sentence.lower()), self.vocab_size, self.n_past_words, self.vocab_path, None, silent)

        # start tensorflow session
        sess = tf.Session()

        # load saved session
        checkpoint_file = tf.train.latest_checkpoint(self.storage_dir + '/')
        saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
        saver.restore(sess, checkpoint_file)

        # load input words and predictions from the saved session
        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        predictions = graph.get_operation_by_name("predictions").outputs[0]
        
        # get features and predicted pos ids
        features = self.__rnn_reshape(self.data.features) if self.architecture == 'RNN' else self.data.features
        predicted_pos_ids = sess.run(predictions, feed_dict={input_x: features})

        # create lists of the sentence words and their corresponding predicted POS tags
        words = []
        for sentence_word_ids in self.data.features:
            word_id = sentence_word_ids[0]
            words.append(self.data.id_to_word[word_id])
        predicted_pos = []
        for pred_id in predicted_pos_ids:
            predicted_pos.append(self.data.id_to_pos[pred_id])

        # pretty print tagged sentence if enabled
        if pretty_print:
            table = Texttable(200)
            table.set_deco(Texttable.HEADER)
            table.set_cols_dtype(['t' for i in range(len(words))])
            table.set_cols_align(['c' for i in range(len(words))])
            table.add_rows([sentence.split(), predicted_pos], header=False)
            print(table.draw())

        # merge word and tag lists
        word_pos_tuples = zip(words, predicted_pos)
        annotated_words = []
        for tup in word_pos_tuples:
            annotated_word = '%s/%s' % (tup[0], tup[1])
            annotated_words.append(annotated_word)

        return ' '.join(annotated_words)


    def evaluate(self, evaluation_file, print_inline=False):
        """
        Evaluates a previously trained model with a given evaluation file
        """

        # check if evaluation file exists
        if not os.path.isfile(evaluation_file):
            print('Error: evaluation file "%s" doesn\'t exist' % evaluation_file)
            return

        # initialize counter variables
        n_sentences_correct = n_words_correct = n_tags_correct = 0
        tags_wrong = {}
        true_tags, predicted_tags = [], []
        # get pre-tagged evaluation data
        text = open(evaluation_file, encoding="utf8").read()
        text = '\n'.join([s for s in text.splitlines() if s and s[0] != '#'])
        true_sentences = text.splitlines()
        n_sentences = len(true_sentences)
        # tag data based on trained language model
        predicted_words = self.tag(self.__untag_text(text), False, True).split()
        predicted_sentences = []
        sentence_lengths = [len(x.split()) for x in true_sentences]
        n_words = sum(sentence_lengths)
        for i, l in enumerate(sentence_lengths):
            predicted_sentences.append(' '.join(predicted_words[sum(sentence_lengths[:i]):sum(sentence_lengths[:i])+l]))

        # start evaluation
        if not print_inline:
            print('Evaluation starts...')

        # compute correct sentences and words
        for t, p in zip(true_sentences, predicted_sentences):
            t, p = t.strip(), p.strip()
            # check sentences
            if t == p:
                n_sentences_correct += 1
                n_words_correct += len(t.split())
                n_tags_correct += len(t.split())
            else:
                # check true and predicted words and tags
                for tw, pw in zip (t.split(), p.split()):
                    if tw != pw:
                        # build a custom key: correct_word/tag|wrong_word/tag to count its occurence
                        key = tw + '|' + pw
                        tags_wrong[key] = tags_wrong[key] + 1 if key in tags_wrong and tags_wrong[key] > 0 else 1
                    # increment counter for matching words and tags
                    n_words_correct += 1 if tw[:tw.index('/')] == pw[:pw.index('/')] else 0
                    n_tags_correct += 1 if tw[tw.index('/')+1:] == pw[pw.index('/')+1:] else 0
                    # add tag ids to true and predicted tag lists for cohens kappa
                    true_tags.append(self.data.pos_to_id[tw[tw.index('/')+1:]]) 
                    predicted_tags.append(self.data.pos_to_id[pw[pw.index('/')+1:]])
        # calculate cohen's kappa
        kappa = cohen_kappa_score(true_tags, predicted_tags)

        if print_inline:
            print('%s: %i/%i (%.1f%%) tags correct, %.3f kappa score' % (evaluation_file, n_tags_correct, n_words, n_tags_correct/n_words*100, kappa))
        else:
            # print ratio of correct sentences, words and tags
            print('\n# RESULTS:\n')
            table = Texttable(120)
            table.set_deco(Texttable.HEADER)
            table.set_cols_dtype(['t', 'f', 't'])
            table.set_cols_align(['r', 'c', 'l'])
            table.add_row([str(n_sentences_correct) + ' / ' + str(n_sentences), n_sentences_correct/n_sentences, 'sentences correct'])
            table.add_row([str(n_words_correct) + ' / ' + str(n_words), n_words_correct/n_words, 'words recognized'])
            table.add_row([str(n_tags_correct) + ' / ' + str(n_words), n_tags_correct/n_words, 'tags correct'])
            table.add_row(['', kappa, 'kappa score'])
            print(table.draw())
            # show wrong tags
            print('\n# ERRORS:\n')
            table = Texttable(120)
            table.set_deco(Texttable.HEADER)
            table.set_cols_dtype(['i', 't', 't'])
            table.set_cols_align(['c', 'l', 'l'])
            table.set_chars(['-', '|', '+', '-'])
            table.add_rows([["count", "expected", "computed"]])
            for key, count in sorted(tags_wrong.items(), key=lambda x: x[1], reverse=True):
                table.add_row([count, key[:key.index('|')], key[key.index('|')+1:]])
            print(table.draw())

    def reset(self, force=False, quiet=False):
        """
        Executes a reset by deleting all training and logging data
        """

        answer = 'yes' if force else input('Really delete all training data and log files? [Yes/no] ')
        if answer.lower() == 'yes' or answer == '':
            # delete storage files
            self.__empty_directories(self.storage_dir)
            # delete log files
            self.__empty_directories(self.log_dir)
            if not quiet:
                print('Reset was executed. All files successfully deleted.')
        else:
            if not quiet:
                print('Reset was not executed.')


    def __empty_directories(self, directory):
        """
        Delete all files and folders in given directory

        @param directory: directory to delete all files and subdirectories from
        """

        for f in os.listdir(directory):
            if f == '.gitignore':
                continue
            p = os.path.join(directory, f)
            if os.path.isfile(p):
                os.unlink(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)


    def __load_data(self, training_file_path):
        """
        Loads and processes training data from a given training file

        @param training_file_path: Path to a corpus file with tagged sentences of format: word1/TAG word2/TAG with one sentence per line

        @return train_batches: training data splitted into iterable batches
        @return test_data: data to test the trained model
        @return n_pos_tags: total number of existing POS tags
        """

        # read tagged data from training file
        print('Loading training data from "%s"...' % training_file_path)
        with open(training_file_path, 'r', encoding="utf8") as f:
            tagged_sentences = f.read()
            f.close()

        data = TextLoader(tagged_sentences, self.vocab_size, self.n_past_words, self.vocab_path, self.tensor_path)
        x = data.features
        y = data.labels
        n_pos_tags = len(data.pos_to_id)

        # split data for training and testing according to the test ratio
        idx = int(self.test_ratio * len(x))
        x_test, x_train = x[:idx], x[idx:]
        y_test, y_train = y[:idx], y[idx:]

        # create iterable training batches
        if self.architecture == 'RNN':
            train_batches = self.__batch_iterator(list(zip(x_train, y_train)), self.n_epochs, shuffle=False)
        else:
            train_batches = self.__batch_iterator(list(zip(x_train, y_train)), self.n_epochs, shuffle=True)
        test_data = {'x': x_test, 'y': y_test}

        return (train_batches, test_data, n_pos_tags)


    def __batch_iterator(self, data, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset
        """

        data = np.array(data)
        data_size = len(np.atleast_1d(data))
        num_batches_per_epoch = int((data_size-1)/self.batch_size) + 1
        for _ in range(num_epochs):
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


    def __model_init(self, n_pos_tags):
        """
        Initializes a Feed-Forward Neural Network model
        """

        # load model architecture based on settings, default is FNN (Feed-forward Neural Network)
        if self.architecture == 'RNN':
            nn_model = model.RNN(self.h_size, n_pos_tags, self.n_timesteps)
        else:
            nn_model = model.FNN(self.vocab_size, self.n_past_words, self.embedding_size, self.h_size, n_pos_tags)
        global_step = tf.Variable(initial_value=0, name="global_step", trainable=False)
        optimizer = nn_model.optimizer
        train_op = optimizer.minimize(nn_model.loss, global_step=global_step)

        return nn_model, train_op, global_step


    def __logging_init(self, nn_model, graph):
        """
        Set up logging that the progress can be visualised in TensorBoard
        """

        # Add ops to record summaries for loss and accuracy...
        train_loss = tf.summary.scalar("train_loss", nn_model.loss)
        train_accuracy = tf.summary.scalar("train_accuracy", nn_model.accuracy)
        # ...then merge these ops into one single op so that they easily be run together
        train_summary_ops = tf.summary.merge([train_loss, train_accuracy])
        # Same ops, but with different names, so that train/test results show up separately in TensorBoard
        test_loss = tf.summary.scalar("test_loss", nn_model.loss)
        test_accuracy = tf.summary.scalar("test_accuracy", nn_model.accuracy)
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

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        return saver


    def __step(self, sess, model, standard_ops, train_ops, test_ops, x, y, summary_writer, train):
        """
        Execute one training step
        """

        feed_dict = {model.input_x: x, model.input_y: y}

        if train:
            step, loss, accuracy, _, summaries = sess.run(standard_ops + train_ops, feed_dict)
            print("Step %d: loss %.1f, accuracy %d%%" % (step, loss, 100 * accuracy), end="\r", flush=True)
        else:
            step, loss, accuracy, summaries = sess.run(standard_ops + test_ops, feed_dict)
            print("Step %d: loss %.1f, accuracy %d%%" % (step, loss, 100 * accuracy), end="", flush=True)

        summary_writer.add_summary(summaries, step)


    def __replace(self, sentence, replacement_file=conf.REPLACEMENT_FILE, is_tagged=False):
        """
        Replaces synonyms with uniform description
        """

        # check if replacement file exists
        if not os.path.isfile(replacement_file):
            print('Error: replacement file "%s" doesn\'t exist' % replacement_file)
            return

        # create replacement mappings
        replacements = []
        with open(replacement_file, 'r', encoding="utf8") as f:
            for line in f.readlines():
                assignments = line.strip().split('=')
                synonyms, key = assignments[0], assignments[1]
                for synonym in synonyms.split(','):
                    replacements.append([synonym, key])
        # replace occurences in sentence
        replaced_sentence = []
        for word in sentence.split():
            found = False
            for synonym, key in replacements:
                if word == synonym:
                    replaced_sentence.append(key)
                    found = True
                    break
            if not found:
                replaced_sentence.append(word)
        
        return ' '.join(replaced_sentence)


    def __untag_sentence(self, sentence):
        """
        Removes tags from a tagged sentence
        """

        words = []
        for word in sentence.split():
            words.append(word[:(word.index('/'))])
        return ' '.join(words)


    def __untag_text(self, text):
        """
        Removes tags from a tagged text (one sentence per line)
        """

        sentences = []
        for sentence in text.splitlines():
            sentences.append(self.__untag_sentence(sentence))
        return '\n'.join(sentences)


    def __rnn_reshape(self, flat_batches):
        """
        reshapes flat batch list of shape [batch_size, 1] to shape [batch_size, n_timesteps, 1]
        """

        batches = []
        for i, flat_batch in enumerate(flat_batches):
            batch = []
            for t in range(self.n_timesteps):
                # if no predecessors are available (at the beginning of the list), add the firsts element
                if t <= i:
                    batch.append(flat_batches[i-t])
                else:
                    batch.append(flat_batches[0])
            batches.append(batch)
        return batches


# only execute training when file is invoked as a script and not just imported
if __name__ == "__main__":
    # get script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="Invokes training of a language model on given corpus")
    parser.add_argument("--tag", type=str, help="Tags a given sentence with the pretrained language model")
    parser.add_argument("--evaluate", type=str, help="Evaluates pretrained language model with a given evaluation file")
    parser.add_argument("--reset", action='store_true', help="Removes all stored training and log data")

    parser.add_argument("-p", "--pastwords", type=int, help="Number of preceding words to take into account")
    parser.add_argument("-e", "--embeddingsize", type=int, help="Dimension of the word embeddings")
    parser.add_argument("-s", "--hiddensize", type=int, help=" Dimension of the hidden layer")
    parser.add_argument("-n", "--nepochs", type=int, help="Number of training epochs")
    parser.add_argument("-t", "--timesteps", type=int, help="Number of past trained words")

    parser.add_argument("-f", "--force", action='store_true', help="Force operation without confirmation")
    parser.add_argument("-q", "--quiet", action='store_true', help="No output messages")
    parser.add_argument("-i", "--inline", action='store_true', help="Only one line output")
    
    args = parser.parse_args()

    # invoke training
    if args.train is not None:
        # create tagger instance
        if args.pastwords is not None and args.embeddingsize is not None and args.hiddensize is not None and args.nepochs is not None:
            t = Tagger('FNN', n_past_words=args.pastwords, embedding_size=args.embeddingsize, h_size=args.hiddensize, n_epochs=args.nepochs)
        elif args.timesteps is not None and args.hiddensize is not None and args.nepochs is not None:
            t = Tagger('RNN', n_timesteps=args.timesteps, h_size=args.hiddensize, n_epochs=args.nepochs)
        else:
            t = Tagger()
        try:
            t.train(args.train)
        except KeyboardInterrupt:
            print('\nFinished training earlier.', flush=True)
    # invoke tagging of a given sentence
    if args.tag is not None:
        # create tagger instance
        if args.pastwords is not None and args.embeddingsize is not None and args.hiddensize is not None and args.nepochs is not None:
            t = Tagger('FNN', n_past_words=args.pastwords, embedding_size=args.embeddingsize, h_size=args.hiddensize, n_epochs=args.nepochs)
        elif args.timesteps is not None and args.hiddensize is not None and args.nepochs is not None:
            t = Tagger('RNN', n_timesteps=args.timesteps, h_size=args.hiddensize, n_epochs=args.nepochs)
        else:
            t = Tagger()
        print('The tagged sentence is:')
        t.tag(args.tag, True, True)
    # invoke evaluation
    if args.evaluate is not None:
        # create tagger instance
        if args.pastwords is not None and args.embeddingsize is not None and args.hiddensize is not None and args.nepochs is not None:
            t = Tagger('FNN', n_past_words=args.pastwords, embedding_size=args.embeddingsize, h_size=args.hiddensize, n_epochs=args.nepochs)
        elif args.timesteps is not None and args.hiddensize is not None and args.nepochs is not None:
            t = Tagger('RNN', n_timesteps=args.timesteps, h_size=args.hiddensize, n_epochs=args.nepochs)
        else:
            t = Tagger()
        if args.inline:
            t.evaluate(args.evaluate, True)
        else:
            t.evaluate(args.evaluate)
    # invoke reset of training data
    if args.reset:
        # create tagger instance
        t = Tagger()
        t.reset(force=args.force, quiet=args.quiet)