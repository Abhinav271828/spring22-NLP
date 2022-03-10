---
title: Introduction to NLP (CS7.401)
subtitle: |
          | Spring 2022, IIIT Hyderabad
          | 10 Mar, Tuesday (Lecture 12)
author: Taught by Prof. Manish Shrivastava
---

# Neural Network Language Models
## Recurrent Neural Networks (contd.)
We have seen that one issue with RNNs is the vanishing gradient problem.

### Conditional RNNs

![Conditional RNNs](cond.png)

## Long Short-Term Memory Networks

![LSTM Architecture](lstm.png)

The core idea is that $c_t$ (the *cell state*) is changed slowly and information can flow along it unaltered. It can be used as a "context-storer" to retrieve necessary information from at any step.

![Cell State of LSTMs](cellstate.png)
![Output of LSTMs](output.png)