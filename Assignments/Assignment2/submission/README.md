---
title: Introduction to NLP (CS7.401)
subtitle: |
          | Spring 2022, IIIT Hyderabad
          | Assignment 2
author: Abhinav S Menon
numbersections: true
---

# The Code
There are two source code files: `svd.py` and `cbow.py`. They implement classes containing the models for their respective algorithms.  

During running, other files are generated: `vocab.txt` (which contains a list of all the words in the vocabulary); `tokenised.txt` (which contains the sentences in the corpus, tokenised into Python lists); and `*_vectors.txt` (which contains the final embeddings).  

The code assumes that the data is present in a directory above the current one, *i.e.*, under the filename `../data/Electronics_5.json`.

# Execution
In order to execute the models, simply run them using `python3`. For example,

```
python3 svd.py
```

The command line output shows the stages of processing it has reached. In addition, all passes through the corpus print the line number every 1000 lines.  

The `svd.py` file writes the embeddings to `svd_vectors.txt`, while the `cbow.py` file writes them to `cbow_vectors.txt`.

# Restoring the Pre-Trained Model
When `cbow.py` is run, the model is trained, pickled, and stored in a file named `model_pkl`.
