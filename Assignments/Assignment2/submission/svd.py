from re import A
import scipy as sp
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
import numpy as np
import re

STOP_AT = 100000

class SVD:
    def __init__(self, window_size, cutoff_sv):
        self.window_size = window_size
        self.cutoff_sv = cutoff_sv
        self.matrix = []
        self.vocab = set()
        self.word_indices = {}
        self.index_words = {}
        self.dimension = 0
        self.vectors = []

    def get_vocab(self, trainfile="../data/Electronics_5.json"):
        with open(trainfile, "r") as f:
            with open("tokenised.txt", "w") as tok:
                c = 0
                for line in f:
                    c += 1
                    words = word_tokenize(eval(line)["reviewText"])
                    words = [word.lower() for word in words]
                    tok.write(str(words) + '\n')
                    [self.vocab.add(word.lower()) for word in words]
                    if (c % 1000 == 0):
                        print("line number", c)
                    if (c == STOP_AT):
                        break

        self.vocab = list(self.vocab)

        vocab_size = len(self.vocab)
        self.matrix = sp.sparse.lil_matrix((vocab_size, vocab_size))

        with open("vocab.txt", "w") as f:
            for index, word in enumerate(self.vocab):
                self.word_indices.update({word: index})
                self.index_words.update({index: word})
                f.write(word + '\n')

    def get_matrix(self):
        c = 0
        with open("tokenised.txt", "r") as f:
            for line in f:
                c += 1
                words = eval(line)

                window = words[:self.window_size]
                if (len(words) > self.window_size):
                    for t, word in enumerate(words[:-self.window_size]):
                        i = self.word_indices[word]
                        window = window[1:] + [words[t+self.window_size]]
                        for other_word in window:
                            j = self.word_indices[other_word]
                            try:
                                self.matrix[i,j] += 1
                            except KeyError:
                                self.matrix[i,j] = 1
                            except KeyError:
                                self.matrix[i,j] = 1
                            try:
                                self.matrix[j,i] += 1
                            except KeyError:
                                self.matrix[j,i] = 1
                            except KeyError:
                                self.matrix[j,i] = 1

                else:
                    #print("Too short sentence:", len(words))
                    pass

                if (c % 1000 == 0):
                    print("line number", c)
                if (c == STOP_AT):
                    break

    def get_vectors(self):
        u, s, _ = sp.sparse.linalg.svds(self.matrix)

        denominator = sum(s)
        cont_sum = 0
        for i, x in enumerate(s):
            cont_sum += x
            if (cont_sum/denominator > self.cutoff_sv):
                self.dimension = i+1
                break
        self.vectors = u[:, :self.dimension]

        self.vectors = [v/norm(v) for v in self.vectors]

        with open("svd_vectors.txt", "w") as f:
            for word, vector in zip(self.vocab, self.vectors):
                f.write(word + '\n')
                f.write(str(vector) + '\n')
    
    def get_top_ten(self, word):
        vec_words = []

        line1, line2 = (1, 1)
        with open("svd_vectors.txt", "r") as f:
            for line1 in f:
                line1 = line1[:-1]
                line2 = f.readline()
                line2 = re.sub(r' +', ',', line2).replace('[,', '[')
                try:
                  line2 = eval(line2)
                  print("word", line1, "vector", line2)
                  vec_words += [(line1, line2)]
                  if (line1 == word):
                      vector = line2
                except:
                  f.readline()

                #line1 = f.readline()
                #line2 = f.readline()

        sort_v = sorted(vec_words, key=lambda p: np.dot(p[1], vector))
        print("top ten", sort_v[:10])

print("Creating model")
svd_vectors = SVD(2, 0.5)
print("Getting vocab")
svd_vectors.get_vocab()
print("Getting counts")
svd_vectors.get_matrix()
print("Getting vectors")
svd_vectors.get_vectors()
