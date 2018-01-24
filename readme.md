# Master Thesis
**Part-of-Speech Tagging with Neural Networks for a conversational agent**

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

	$ pip install tensorflow

If the installation was successful, change to the directory of the Tagger you want to use and everything should be ready to run properly; e.g. the Feed-forward Neural Network Tagger:

	$ cd fnn-tagger/

## Usage
### Configuration
TODO

### Training
To train the Tagger call the `tagger.py` script with the `--train` flag:

	$ python tagger.py --train

According to your configuration, the batch training will start. Once you reached a sufficient accuracy, you can interrupt the training or wait till the end. An example output would be:

	Training starts...
	Loading training data from "data/test.corpus"...
	Generating vocabulary...
	Generating tensors...
	Initializing model...
	Step 100: loss 0.9, accuracy 91%
	Saved model checkpoint to 'saved/model-100'
	Step 200: loss 0.2, accuracy 98%
	Saved model checkpoint to 'saved/model-200'
	Step 300: loss 0.0, accuracy 100%
	Saved model checkpoint to 'saved/model-300'

### Tagging
To tag a sentence with a pretrained model call the `tagger.py` script with the `--tag` parameter with a sentence to be tagged:

	$ python tagger.py --tag "Show all modules of Bachelor Informatics"

Now all words are looked up in the model and get a tag, e.g.:

	Show/R_LIST all/X modules/M_MTSModule of/X Bachelor/C_Program:degree Informatics/C_Program:name

## Links
### Alex
- https://gitlab.tubit.tu-berlin.de/thilo.michael/aaca-alex/wikis/home
- https://gitlab.tubit.tu-berlin.de/thilo.michael/aaca-alex/wikis/documentation-tagging

### TensorFlow
- https://www.tensorflow.org/get_started/mnist/pros
- https://www.tensorflow.org/versions/master/api_docs/python/tf/nn

### Similar Tagger
- https://github.com/mrahtz/tensorflow-pos-tagger