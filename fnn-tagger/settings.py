"""
Settings for the Tagger
"""

import os

# ingore tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Path to a file with tagged sentences of this form: word1/TAG word2/TAG ...
TRAINING_FILE_PATH = 'data/test.corpus'

# Presumably dimension of the vocabulary (number of distinct words)
VOCAB_SIZE = 15000

# Number of preceding words to take into account for the POS tag training of the current word
N_PAST_WORDS = 3

# Dimension of the word embeddings
EMBEDDING_SIZE = 50

# Dimension of the hidden layer
HIDDEN_LAYER_SIZE = 100

# Ratio of test data extracted from the training data
TEST_RATIO = 0.1

# Size of the training batches
BATCH_SIZE = 64

# Number of training epochs
N_EPOCHS = 100

# Show evaluation result after this number of trainings steps
EVALUATE_EVERY = 100

# Save model state after this number of trainings steps
CHECKPOINT_EVERY = 100
