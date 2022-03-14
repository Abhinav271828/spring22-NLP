---
title: Introduction to NLP (CS7.401)
subtitle: |
          | Spring 2022, IIIT Hyderabad
          | 11 Mar, Friday (Lecture 13)
author: Taught by Prof. Manish Shrivastava
---

# Neural Network Language Models
## Long Short-Term Memory Networks
A modified version of LSTMs, the GRU (gated recurrent unit), merges the cell state $c_t$ and hidden state $h_t$. It also combines the *forget* and *input* gates into a single *update* gate.  
GRUs ensure that $h_t$ either retains a high amount of old information or starts over with a high amount of new information.

![Gated Recurrent Unit](gru.png)

## Applications
### Sequence to Sequence Chat Model

![Sequence to Sequence Chat Model using LSTMs](seq.png)

### Neural Machine Translation

![Neural Machine Translation using LSTMs](nmt.png)