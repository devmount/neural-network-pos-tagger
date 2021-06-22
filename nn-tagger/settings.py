"""
Settings for the Tagger
"""

import os

# ingore tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# architecture: FNN|RNN
ARCHITECTURE = 'FNN'

# Presumable dimension of the vocabulary (number of distinct words)
VOCAB_SIZE = 9142

# Number of preceding words to take into account for the POS tag training of the current word [FNN only]
N_PAST_WORDS = 3

# Number of previous training steps to take into account [RNN only]
N_TIMESTEPS = 3

# Dimension of the word embeddings [FNN only]
EMBEDDING_SIZE = 100

# Dimension of the hidden layer
HIDDEN_LAYER_SIZE = 100

# Ratio of test data extracted from the training data
TEST_RATIO = 0.1

# Size of the training batches
BATCH_SIZE = 32

# Number of training epochs
N_EPOCHS = 20

# Show evaluation result and save model state after this number of trainings steps
CHECKPOINT_EVERY = 100

# replacement file path
REPLACEMENT_FILE = 'data/replacement.txt'
