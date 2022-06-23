# Abstract
- neural dependency parser
- LSTM: part of speech, labelled dep parses
- rare word problem: char-based word rep
    - LSTM to produce embeddings

# Introduction
- builds on deep biaffine neural dependency parser
    - Dozat and Manning (2017)
- incorporate word rep for rare words
- train taggers

# Architecture
## Deep Biaffine Parser
- D&M
- input: sequence of tokens + PoS tags
- multilayer bidirectional LSTM network
- output state (excluding cell state) through four ReLU layers
    -> 4 vector reps
        - dep seeking head
        - head seeking deps
        - dep seeking label
        - head seeking deps' labels

- two biaffine classifiers
    - score for each (tok1,tok2)
    - score for each ((tok1,tok2),label)
- trained by optimising sum of softmax cross-entropy loss

- Formalism

## Character-Level Model
- D&M: pretrained vector + holistic embedding
- + third <- char seq
- Formalism

- then add with pre-trained embedding and holistic freq-tok embedding
- also add embeddings for UPOS and XPOS

- these are inputs to 2.1 for BiLSTM

## PoS Tagger
- again BiLSTM over word vectors
- ReLU for one vector rep for each tag

# Training Details
- three BiLSTM layers
- parser
    - 100D word, tag
    - 200D recurrent states
- arc classifier
    - 400D head/dep vector
- label classifier
    - 100D

- drop word & tag independently with 1/3 chance; scale the other
- something and all

- learning rate 0.002
- β1 = β2 = 0.9

- save model every 100 steps (5k words each) if <1k iters
- then only if accuracy increases, until 5k steps

- character moel
    - 100D char embeddings
    - 400D recurrent states
    - no dropping

- tagger
    - two BiLSTM layers 
    - dropout 50%

# Results
