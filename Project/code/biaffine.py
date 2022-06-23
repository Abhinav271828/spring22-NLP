import numpy as np

class ArcBiaff:
  def __init__(self, n_words, wordvec_dim=400):
    self.w = np.random.rand(n_words, wordvec_dim)
    self.b = np.random.rand(1, n_words)
    self.sentences = []

  def get_training_data(self, train_file):
    sentence = []
    sentence_dict = {0: "root"}

    line = "\n"
    with open(train_file, "r") as f:
      for line in f:
        if (line[0] == '#'):
          continue
        if (len(line) == 1):
          sentence = [(word, sentence_dict[head_index])
                        for (word, head_index) in sentence]
          self.sentences += [sentence]
          sentence = []
          sentence_dict = {0: "root"}
        else:
          columns = line.split('\t')
          sentence_dict.update({int(columns[0]): columns[1]})
          sentence.append((columns[1], int(columns[6])))
