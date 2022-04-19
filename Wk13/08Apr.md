---
title: Introduction to NLP (CS7.401)
subtitle: |
          | Spring 2022, IIIT Hyderabad
          | 08 Apr, Friday (Lecture 19)
author: Taught by Prof. Manish Shrivastava
---

# BERT
BERT has a transformer architecture; more accurately, BERT is the encoder stack of a transformer.  

We have seen how transformers make use of the context information using multiple passes through the encoder, including positional knowledge. Refer: [Illustrated BERT](http://jalammar.github.io/illustrated-bert/)  

The problem with previous methods is that language understanding is bidirectional. BERT, however, takes information for each word from *all* the words in the sentence; it is $n$-directional.  

It uses a pseudo-token `[CLS]`, which in theory is responsible for *collecting* the information from all the words in the input.  

Since BERT is not made for any single downstream task, we need a task to train it on. We use a CBOW-type language modelling task (differing from ordinary CBOW in that context size is variable). 15% of the tokens in the sentence are randomly chosen to be predicted; about 80% of them are replaced with `[MASK]`, and the remaining are replaced with random noise words.  
We might also use next-sentence prediction to train BERT.  

This training allowed BERT to solve on a wide variety of problems, *e.g.*, sentence pair classification, single-sentence classification, question-answering, and so on.  

BERT uses a variant of the WordPiece model, which minimises vocabulary size by splitting unknown words into smaller parts (*byte-pairs*). This allows it to use a relatively small vocabulary of just 30,000 words.

Architecturally, BERT is a pretrained 12-layer (BERT Large has 24-layers) transformer encoder stack.  
When BERT is to be used on a downstream task (like sentence classification), its output is passed through a *head* (an MLP), whose output is used to perform the task. This is called *fine-tuning*.  

Refer: Analysis of learnings by layers of BERT (Ganesh Jawahar), BERTology (Anna Rogers)

# Post-BERT
RoBERTa is a model with BERT's architecture, trained on more data for more epochs. It slightly improved the performance of BERT.  

XLNet had two major innovations: *relative* positional encoding (the positional encoding depends on the focus word) and permutational language modelling. However, analysis showed that these innovations did not change a lot in the model's results.  

ALBERT is a lighter version of BERT that changes its internal representations; it uses smaller embedding sizes projected to a larger transformer hidden size. Furthermore, it shares all parameters between transformer layers. It reduces the cost of BERT (in terms of computational infrastructure, *not* training time).

# Distillation
Distillation is the probolem of model compression - create a lighter model that preserves the performance of the bigger model. It involves creating a new (much smaller) model (a "student"), and training it to mimic a larger, SOTA-pretrained, fine-tuned model (the "teacher"). The loss function of the student model is typically MSE or cross-entropy. Refer: Well-read Students Learn Better (Turc *et al.* 2020)
