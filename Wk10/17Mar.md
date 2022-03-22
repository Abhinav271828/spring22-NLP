---
title: Introduction to NLP (CS7.401)
subtitle: |
          | Spring 2022, IIIT Hyderabad
          | 17 Mar, Thursday (Lecture 15)
author: Taught by Prof. Manish Shrivastava
---

# Frames
Words have semantic requirements from entities that are associated with them. For example, the verb *flying* cannot take the subject *rock*, as it requires its subject to have agency.

# Machine Translation
We have seen that the meanings of words are influenced by those of the words in their vicinity. In addition to this, our intuition tells us that we do not translate a sentence in one go (like a encoder-decoder LSTM model), but instead, dynamically.  

We would like a model that captures both these features.