# Neural Network POS Tagger

> Part-of-Speech Tagging with Neural Networks for a conversational agent

This toolkit was implemented during my master's thesis, that aimed to improve the natural language understanding of an artificial conversational agent. This agent utilized a Hidden Markov Model to calculate Part-of-Speech tags for input words. In order to achieve better results, two different classification architectures are implemented and evaluated: a Feed-forward Neural Network and a Recurrent Neural Network.

This repository contains the toolkit to train and evaluate language models for POS tagging and tag input sentences according to a trained model. It provides the possibility to use the python scripts directly as well as an API and is licensed under GPL-3.0.

## Setup

Check if Python version >= 3.5 is installed:

    $ python --version
    Python 3.6.3

Install dependencies (consider using a virtual environment):

    pip install -r requirements.txt

If the installation was successful, change to the directory of the Tagger and everything should be ready to run properly:

    cd fnn-tagger/

## Script Usage

The single python scripts of this toolkit can be called directly, documented in the following.

### Configuration

Static settings are located in the `settings.py` script. It contains the following configuration options:

| option | description |
| ------ | ----------- |
| `ARCHITECTURE` | Neural network architecture that will be used. Possible values: `FNN`, `RNN` |
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

However, the following sections explain the usage of the specific flags and their apropriate combination for each action.

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

You can also call the script with inline configuration. To train a model using the FNN architecture, use the flags `-p`, `-e`, `-s`, and `-n`. It is required to use exactly these 4 flags, otherwise the static configuration from the `settings.py` will be used. An example call would be:

    python tagger.py --train data/test.corpus -p 1 -e 250 -s 350 -n 5

To train a model using the RNN architecture, use the flags `-t`, `-e`, `-s`, and `-n`. It is required to use exaclty these 4 flags, otherwise the static configuration from the `settings.py` will be used. An example call would be:

    python tagger.py --train data/test.corpus -t 8 -e 100 -s 100 -n 5

### Tagging

To tag a sentence with a pretrained model call the `tagger.py` script with the `--tag` parameter followed by sentence to be tagged. Now a tag is attached to every word.

    $ python tagger.py --tag "Show all modules of Bachelor Informatics"
    The tagged sentence is:
     Show     all       modules     of       Bachelor         Informatics
    R_LIST   R_LIST   M_MTSModule    X    C_Program:degree   C_Program:name

Make sure that the `settings.py` is configured with the same values that were used to train the model, otherwise the tagger cannot load the pretrained model correctly.

If you don't want to be bothered by the `settings.py`, you can also call the script with inline configuration. To tag a sentence using the FNN architecture, use the flags `-p`, `-e` and `-s`. It is required to use exactly these 3 flags, otherwise the static configuration from the `settings.py` will be used. An example call would be:

    python tagger.py --tag "Show all modules of Bachelor Informatics" -p 1 -e 250 -s 350

To tag a sentence using the RNN architecture, use the flags `-t`, `-e` and `-s`. It is required to use exactly these 3 flags, otherwise the static configuration from the `settings.py` will be used. An example call would be:

    python tagger.py --tag "Show all modules of Bachelor Informatics" -t 8 -e 100 -s 100

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
                0.966   kappa score

    # ERRORS:

    count           expected                    computed
    ------------------------------------------------------------
    4     bachelor/C_Program:degree   bachelor/C_Program:name
    4     master/C_Program:degree     master/C_Program:name
    1     institute/X_Chair:name      institute/C_Program:name
    1     quality/C_Chair:name        quality/C_Program:name
    1     and/C_Chair:name            <UNKNOWN_WORD>/X

 Make sure that the `settings.py` is configured with the same values that were used to train the model, otherwise the evaluation cannot load the pretrained model.

 If you don't want to be bothered by the `settings.py`, you can also call the script with inline configuration. To tag a sentence using the FNN architecture, use the flags `-p`, `-e` and `-s`. It is required to use exactly these 3 flags, otherwise the static configuration from the `settings.py` will be used. An example call would be:

    python tagger.py --evaluate data/evaluation.txt -p 1 -e 250 -s 350

To tag a sentence using the RNN architecture, use the flags `-t`, `-e` and `-s`. It is required to use exactly these 3 flags, otherwise the static configuration from the `settings.py` will be used. An example call would be:

    python tagger.py --evaluate data/evaluation.txt -t 8 -e 100 -s 100

If you don't need the list of errors in the evaluation result, you can also print it in one line, adding the `-i` flag, i.e.:

    $ python tagger.py --evaluate data/evaluation.txt -p 1 -e 250 -s 350 -i
    data/evaluation_known.txt: 197/208 (94.7%) tags correct, 0.966 kappa score

### Reset

To reset the tagger and delete all previously created files call the `tagger.py` script with the `--reset` flag and confirm with 'Yes' (or <kbd>Enter</kbd>):

    $ python tagger.py --reset
    Really delete all training data and log files? [Yes/no]
    Reset was executed. All files successfully deleted.

If you don't want to be bothered by a security question, you can use the `-f` flag to force a direct deletion:

    python tagger.py --reset -f

If you even don't want to be bothered by any output messages, you can use the `-q` flag to force a quiet deletion:

    python tagger.py --reset -q

## API Usage

This toolkit is designed to be used in other applictions. The API is documented in the following.

### Configuration

Import the tagging script properly according to your directory structure, i.e.:

    import tagger as nn

To instantiate the tagger, just call the `Tagger()` class. Without any parameters, the static configuration from the `settings.py` will be used:

    t = nn.Tagger()

If you prefer inline configuration, pass the corresponding parameters according to the neural network architecture (the FNN needs `n_past_words`, the RNN needs `n_timesteps`). See these two examples, one for each architecture:

    t = nn.Tagger('FNN', n_past_words=1, embedding_size=250, h_size=350, n_epochs=5)
    t = nn.Tagger('RNN', n_timesteps=8, embedding_size=100, h_size=100, n_epochs=5)

### Training

To train the initialized tagger, just call the `train()` method with the path to the corpus file, i.e.:

    t.train('data/test.corpus')

The trained model will be stored in the `saved/` directory.

### Tagging

A sentence can by tagged with a pretrained model by calling the `tag()` method. You have additional parameters to print the tagging output in tabular form to the console (`pretty_print`) or mute console messages concerning model loading completely (`silent`).

    tagged_sentence = t.tag('Show all modules of Bachelor Informatics', format_list=False, pretty_print=True, silent=False)

If you want to process lists instead of strings, you can use `format_list=True` as parameter. Now the input sentence has to be a list of words, an the output will be a list of word tag tuples:

    $ sentence = ['Show', 'all', 'modules', 'of', 'Bachelor', 'Informatics']
    $ tagged_sentence = t.tag(sentence, format_list=True, pretty_print=False, silent=True)
    $ print list(tagged_sentece)
    [('Show', 'R_LIST'), ('all', 'R_LIST'), ('modules', 'M_MTSModule'), ('of', 'X'), ('Bachelor', 'C_Program:degree'), ('Informatics', 'C_Program:name')]

### Evaluation

To evaluate a pretrained model, calle the `evaluate()` method. You have an additional parameter to print the main evaluation results in one single line.

    t.evaluate('data/evaluation.txt', print_inline=False)

## Resources

### TensorFlow

- <https://www.tensorflow.org/get_started/mnist/pros>
- <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn>

### POS Tagger

- <https://github.com/mrahtz/tensorflow-pos-tagger>
- <https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb>

---

If you like and use this POS tagger and want to give some love back, feel free to...

<p align="center">
  <a href="https://www.buymeacoffee.com/devmount" target="_blank">
  <img alt="Buy me a coffee" src="https://user-images.githubusercontent.com/5441654/44213163-60a91100-a16d-11e8-9d5d-7d862cae7b7c.png">
  </a>
</p>
