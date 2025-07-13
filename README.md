# Foundation models for neuroscience tutorial

Tutorial repository for the Cajal school in machine learning for neuroscience. 

Implements an NDT-1 style encoder from scratch (Ye and Pandarinath, 2021).

## Running the tutorial in colab (recommended)

![Click this link](https://colab.research.google.com/github/patrickmineault/fmn-tutorial/blob/main/tutorials/Tutorial.ipynb)

## Compiling the tutorial from the .py source

`cd` into the tutorials folder, and run:

```
jupytext tutorial_source.py --to ipynb -o Tutorial.ipynb
```

## Working with the scripts

* Fork this repository
* Clone your fork
* Create a fresh conda environment with e.g. Python 3.9, `conda create --name fmn-tutorial python=3.11`
* `cd` into the directory and `pip install -e .`

There are three scripts available in the scripts folder:

* `preprocess_data.py`: creates pickle files from DANDI sources
* `train_autoencoder.py`: trains auto-encoders, similar to the tutorial, but on the command line, with better logging
* `predict_behavior.py`: trains a decoder for behavior from scratch or from a pretrained checkpoint
