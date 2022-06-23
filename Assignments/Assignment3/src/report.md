---
title: Introduction to NLP (CS7.401)
subtitle: |
          | Spring 2022, IIIT Hyderabad
          | Assignment 3
author: Abhinav S Menon
numbersections: true
---

# Language Model
## Model Details
The model contains one embedding layer, an LSTM layer (unidirectional, single layer), and an output layer.  

The model was trained (on each corpus) until the difference in the loss between successive epochs is less than 0.01. The embedding dimension is 100 and the hidden size is 150. The learning rate is 0.001.  

The trained models are saved as `{eng,fr}_lm.pth`.  

## Results
The perplexities of the train and test splits of the Europarl corpus are given in `2020114001_LM_{train,test}.txt`. The averages are approximately 1.3 for both these splits.

## Analysis
The perplexity scores are very good compared to those obtained by statistical methods like $n$-grams. This is most likely due to the availability of an arbitrarily long context; it allows for much better prediction.

# MT Model
## Model Details
The encoder and decoder are both LSTMs with the same hyperparameters as in the LMs. The early stopping condition and the learning rate are also identical.

## Results
The BLEU scores of the model trained from scratch are given in `2020114001_MT1_{train,test}.txt`. The corpus scores are approximately 0.27 on the train split and 0.23 on the test split.

# Directory Structure
```
.
├── README.md
├── report.pdf
├── result
│   ├── 2020114001_LM_test.txt
│   ├── 2020114001_LM_train.txt
│   ├── 2020114001_MT1_test.txt
│   └── 2020114001_MT1_train.txt
├── saved
│   ├── eng_lm.pth
│   └── fr_lm.pth
└── src
    ├── __pycache__
    │   ├── tokenise.cpython-39.pyc
    │   └── tokenize.cpython-39.pyc
    ├── language_model.py
    ├── report.md
    ├── seq2seqmt1.py
    ├── seq2seqmt2.py
    └── tokenise.py
```
