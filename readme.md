# Master Thesis

> Part-of-Speech Tagging with Neural Networks for a conversational agent

This master thesis aims to improve the natural language understanding of an artificial conversational agent, that uses a Hidden Markov Model to calculate Part-of-Speech tags for input words. In order to achieve better results especially for uncommon word combinations and sentence structures, two new classification models are implemented and evaluated: a Feed-forward Neural Network and a Recurrent Neural Network.

## TOC

- Introduction
  - Scope of this Thesis
  - Related Work
    - The Hidden Markov Model
    - The Artificial Neural Network Model
  - Structure of this Thesis
- Alex: Artificial Conversational Agent
  - System Overview
  - Training Data
  - The Hidden Markov Model Tagger
  - Tagging Interface
- Part-of-Speech Tagging with Neural Networks
  - Feed-forward Neural Network Model
    - Architecture
    - Implementation
  - Recurrent Neural Network Model
    - Architecture
    - Implementation
- Training of Language Models
  - Training Data Corpus
  - Parameter Tuning
- Evaluation and Comparison
  - Test Design
  - Evaluation Results
    - Feed-forward Neural Network Models
    - Recurrent Neural Network Models
    - Hidden Markov Models
  - Overall Comparison
- Discussion and Conclusion
  - Summary
  - Discussion
  - Future Work

## Setup

Check if Python version >= 3.5 is installed:

    $ python --version
    Python 3.6.3

Install dependencies:

    pip install tensorflow texttable sklearn scipy

If the installation was successful, change to the directory of the Tagger and everything should be ready to run properly:

    cd fnn-tagger/

## Usage

### Configuration

Static settings are located in the `settings.py` script. It contains the following configuration options:

| option | description |
| ------ | ----------- |
| `ARCHITECTURE` | Neural network architecture that will be used. Possible values: 'FNN', 'RNN' |
| `VOCAB_SIZE` | Setup dimension of the vocabulary |
| `N_PAST_WORDS` | Number of preceding words to take into account for the POS tag training of the current word (FNN only) |
| `N_TIMESTEPS` | Number of previous training steps to take into account (RNN only) |
| `EMBEDDING_SIZE` | Dimension of the word embeddings (FNN only) |
| `H_SIZE` | Dimension of the hidden layer |
| `TEST_RATIO` | Ratio of test data extracted from the training data |
| `BATCH_SIZE` | Size of the training batches |
| `N_EPOCHS` | Number of training epochs |
| `CHECKPOINT_EVERY` | Evaluate and save model state after this number of trainings steps |
| `REPLACEMENT_FILE` | Preprocess training data by normalizing terms with the helo of replacements, stored in this file |

Training, evaluation and tagging can be executed using the `tagger.py` script, which represents the core script of this toolkit. Its general usage is:

    python tagger.py [-h] [--train TRAIN] [--tag TAG] [--evaluate EVALUATE]
                     [--reset] [-p PASTWORDS] [-e EMBEDDINGSIZE] [-s HIDDENSIZE]
                     [-n NEPOCHS] [-t TIMESTEPS] [-f] [-q] [-i]

    optional arguments:
      -h, --help            show this help message and exit
      --train TRAIN         Invokes training of a language model on given corpus
      --tag TAG             Tags a given sentence with the pretrained language
                            model
      --evaluate EVALUATE   Evaluates pretrained language model with a given
                            evaluation file
      --reset               Removes all stored training and log data
      -p PASTWORDS, --pastwords PASTWORDS
                            Number of preceding words to take into account
      -e EMBEDDINGSIZE, --embeddingsize EMBEDDINGSIZE
                            Dimension of the word embeddings
      -s HIDDENSIZE, --hiddensize HIDDENSIZE
                            Dimension of the hidden layer
      -n NEPOCHS, --nepochs NEPOCHS
                            Number of training epochs
      -t TIMESTEPS, --timesteps TIMESTEPS
                            Number of past trained words
      -f, --force           Force operation without confirmation
      -q, --quiet           No output messages
      -i, --inline          Only one line output

However, the following sections explain the usage of the specific flags for each action.

### Training

To train the Tagger call the `tagger.py` script with the `--train` flag. According to your static configuration, the batch training will start. Once you reached a sufficient accuracy, you can interrupt the training with <kbd>CTRL</kbd>+<kbd>C</kbd> or wait till the training process finishes.

    $ python tagger.py --train data/test.corpus
    Training starts...
    Loading training data from "data/test.corpus"...
    Generating vocabulary...
    Generating tensors...
    Initializing model...
    Step 100: loss 0.9, accuracy 91% - saved model checkpoint to 'saved/model-100'
    Step 200: loss 0.2, accuracy 98% - saved model checkpoint to 'saved/model-200'
    Step 300: loss 0.0, accuracy 100% - saved model checkpoint to 'saved/model-300'

You can also call the script with inline configuration. To train a model using the FNN architecture, use the flags `-p`, `-e`, `-s`, and `-n`. It is required to use all 4 flags, otherwise the static configuration from the `settings.py` will be used. An example call would be:

    python tagger.py --train data/test.corpus -p 1 -e 250 -s 350 -n 5

To train a model using the RNN architecture, use the flags `-t`, `-s`, and `-n`. It is required to use all r flags, otherwise the static configuration from the `settings.py` will be used. An example call would be:

    python tagger.py --train data/test.corpus -t 8 -s 50 -n 5

### Tagging

To tag a sentence with a pretrained model call the `tagger.py` script with the `--tag` parameter with a sentence to be tagged. Now a tag is attached to every word.

    $ python tagger.py --tag "Show all modules of Bachelor Informatics"
    The tagged sentence is:
     Show     all       modules     of       Bachelor         Informatics
    R_LIST   R_LIST   M_MTSModule    X    C_Program:degree   C_Program:name

### Evaluation

To evaluate a pretrained model on an external test set call the `tagger.py` script with the `--evaluate` parameter with the path to the file which contains the evaluation data. The evaluation data file must contain one sentence per line, containing space separated word/tag tuples.

    $ python tagger.py --evaluate data/evaluation.txt
    Loading saved vocabulary...
    Generating tensors...
    Evaluation starts...

    # RESULTS:

      20 / 29   0.690   sentences correct
    207 / 208   0.995   words recognized
    197 / 208   0.947   tags correct

    # ERRORS:

    count           expected                    computed
    ------------------------------------------------------------
    4     bachelor/C_Program:degree   bachelor/C_Program:name
    4     master/C_Program:degree     master/C_Program:name
    1     institute/X_Chair:name      institute/C_Program:name
    1     quality/C_Chair:name        quality/C_Program:name
    1     and/C_Chair:name            <UNKNOWN_WORD>/X

 Make sure that the `settings.py` is configured with the same values that were used to train the model, otherwise the evaluation cannot load the pretrained model.

### Reset

To reset the tagger and delete all previously created files call the `tagger.py` script with the `--reset` flag and confirm with 'Yes' (or <kbd>Enter</kbd>):

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

### Example Tagger

- <https://github.com/mrahtz/tensorflow-pos-tagger>
- <https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb>