# Master Thesis

> Part-of-Speech Tagging with Neural Networks for a conversational agent

This master thesis aims to improve the natural language understanding of an artificial conversational agent, that uses a Hidden Markov Model to calculate Part-of-Speech tags for input words. In order to achieve better results especially for uncommon word combinations and sentence structures, two new classification models are implemented and evaluated: a Feed-forward Neural Network and a Recurrent Neural Network.

## TOC (presumably)

- Introduction
  - Scope of this Thesis
  - Related Work
- Alex: Artificial Conversational Agent
  - System Overview
  - Hidden Markov Model
  - Tagging Interface
- Part-of-Speech Tagging
  - Feed-forward Neural Network Model
    - Architecture
    - Implementation
  - Recurrent Neural Network Model
    - Architecture
    - Implementation
- Training
  - Data Retrieval
  - Parameter Tuning
- Evaluation and Comparison
  - Test Design
- Discussion and Conclusion
  - Summary
  - Discussion
  - Future Work

## Setup

Check if Python 3 is installed:

    $ python --version
    Python 3.6.3

Install Tensorflow:

    pip install tensorflow

If the installation was successful, change to the directory of the Tagger you want to use and everything should be ready to run properly; e.g. the Feed-forward Neural Network Tagger:

    cd fnn-tagger/

## Usage

### Configuration

The `settings.py` script contains the following configuration options:

| option | description |
| ------ | ----------- |
| `TRAINING_FILE_PATH` | Path to a file with tagged sentences of this form: word1/TAG word2/TAG ... |
| `VOCAB_SIZE` | Presumable dimension of the vocabulary (number of distinct words) |
| `N_PAST_WORDS` | Number of preceding words to take into account for the POS tag training of the current word |
| `EMBEDDING_SIZE` | Dimension of the word embeddings |
| `H_SIZE` | Dimension of the hidden layer |
| `TEST_RATIO` | Ratio of test data extracted from the training data |
| `BATCH_SIZE` | Size of the training batches |
| `N_EPOCHS` | Number of training epochs |
| `EVALUATE_EVERY` | Show evaluation result after this number of trainings steps |
| `CHECKPOINT_EVERY` | Save model state after this number of trainings steps |

### Training

To train the Tagger call the `tagger.py` script with the `--train` flag. According to your configuration, the batch training will start. Once you reached a sufficient accuracy, you can interrupt the training or wait till the training process finishes.

    $ python tagger.py --train
    Training starts...
    Loading training data from "data/test.corpus"...
    Generating vocabulary...
    Generating tensors...
    Initializing model...
    Step 100: loss 0.9, accuracy 91% - saved model checkpoint to 'saved/model-100'
    Step 200: loss 0.2, accuracy 98% - saved model checkpoint to 'saved/model-200'
    Step 300: loss 0.0, accuracy 100% - saved model checkpoint to 'saved/model-300'

### Tagging

To tag a sentence with a pretrained model call the `tagger.py` script with the `--tag` parameter with a sentence to be tagged. Now a tag is attached to every word.

    $ python tagger.py --tag "Show all modules of Bachelor Informatics"
    Loading saved vocabulary...
    Generating tensors...
    Show/R_LIST all/X modules/M_MTSModule of/X Bachelor/C_Program:degree Informatics/C_Program:name

### Reset

To reset the tagger and delete all previously created files call the `tagger.py` script with the `--reset` flag and confirm with 'Yes':

    $ python tagger.py --reset
    Really delete all training data and log files? [Yes/no]
    Reset was executed. All files successfully deleted.

## Links

### Alex

- <https://gitlab.tubit.tu-berlin.de/thilo.michael/aaca-alex/wikis/home>
- <https://gitlab.tubit.tu-berlin.de/thilo.michael/aaca-alex/wikis/documentation-tagging>

### TensorFlow

- <https://www.tensorflow.org/get_started/mnist/pros>
- <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn>

### Similar Tagger

- https://github.com/mrahtz/tensorflow-pos-tagger
- https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb