"""
Settings for the Tagger
"""

import os

# ingore tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Presumable dimension of the vocabulary (number of distinct words)
VOCAB_SIZE = 15000

# Number of preceding words to take into account for the POS tag training of the current word
N_PAST_WORDS = 3

# Dimension of the word embeddings
EMBEDDING_SIZE = 50

# Dimension of the hidden layer
HIDDEN_LAYER_SIZE = 50

# Ratio of test data extracted from the training data
TEST_RATIO = 0.1

# Size of the training batches
BATCH_SIZE = 32

# Number of training epochs
N_EPOCHS = 10

# Show evaluation result and save model state after this number of trainings steps
CHECKPOINT_EVERY = 500
