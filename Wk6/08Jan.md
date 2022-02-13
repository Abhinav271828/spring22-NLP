---
title: Introduction to NLP (CS7.401)
subtitle: |
          | Spring 2022, IIIT Hyderabad
          | 08 Feb, Monday (Lecture 7)
author: Taught by Prof. Manish Shrivastava
---

# Part-of-Speech Tagging
## Hidden Markov Models (contd.)
### Re-Estimation of Parameters
We wish to adjust $\lambda = (A, B, \pi)$ to maximise $P(O \mid \lambda)$ for some $O$. The Baum-Welch algorithm does this.  

This is an expectation maximisation algorithm.  
It finds the expected number of times the state sequence passes through $s_i$ and $s_j$ at times $t$ and $t+1$. We know that the probability of the transition from $s_i$ to $s_j$ is
$$a_{ij} b_j(o_{t+1}).$$
We can multiply this with $\alpha_t(i)$ (which contains the sum of all paths ending at $s_i$ in the $t^\text{th}$ position) and $\beta_{t+1}(j)$ (which contains the sum of all paths starting from $s_j$ in the $(t+1)^\text{th}$ position) to obtain the overall probability. We call this
$$\begin{split}
\xi_t(i,j) &= P(q_t = s_i, q_{t+1} = s_j \mid O, \lambda) \\
&= \frac{\alpha_t(i) a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)}{P(O \mid \lambda)} \\
&= \frac{\alpha_t(i) a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)}{\sum_{i=1}^N \sum_{j=1}^N {\alpha_t(i) a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)}}. \end{split}$$

We further let
$$\gamma_t(i) = \sum_{j=1}^N \xi_t(i,j),$$
*i.e.*, the number of ways the sequence can pass through $s_i$ at position $t$.  

Therefore the expected number of transitions from $s_i$ is
$$\sum_{t=1}^{T-1} \gamma_t(i)$$
and the expected number of transitions from $s_i$ to $s_j$ is
$$\sum_{t=1}^{T-1} \xi_t(i,j).$$

The Baum-Welch algorithm then assigns
$$\begin{split}
\overline{\pi_i} &= \text{expected frequency of } s_i \text{ at time 1} = \gamma_1(i) \\
\overline{a_{ij}} &= \frac{\text{expected number of transitions from } s_i \text{ to } s_j}{\text{expected number of transitions from } s_i} = \frac{\sum_{t=1}^{T-1} \gamma_t(i)}{\sum_{t=1}^{T-1} \xi_t(i,j)} \\
\overline{b_j(o_k)} &= \frac{\text{expected number of times } s_j \text{ occurs with observation } o_k}{\text{expected number of times } s_j \text{ occurs}} = \frac{\sum_{t=1 \text{with } o_t = o_k}^T \gamma_t(i)}{\sum_{t=1}^T \gamma_t(i)} \end{split}$$

This procedure can be repeated, producing a better model with each iteration.