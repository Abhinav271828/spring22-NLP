from tkinter import W
import torch
from torch import nn, tensor, mean, unsqueeze, cat, transpose, matmul, sigmoid, arange, abs, optim, zeros
import torch.nn.functional as F
from torch.autograd import Variable
from nltk.tokenize import word_tokenize
from numpy import random
import numpy as np
import pickle

STOP_AT = 15

class MeanConcat(nn.Module):
    def __init__(self, ws_into_2):
        super().__init__()
        self.ws_into_2 = ws_into_2
        self.weight = tensor([])

    def forward(self, x):
        mean_first_2m = mean(x[:self.ws_into_2],0)
        remaining = x[self.ws_into_2:]
        result = cat([unsqueeze(mean_first_2m, 0), remaining])
        return result

class GetSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        first_row = x[0:1]
        remaining = x[1:]
        result = matmul(first_row, transpose(remaining,0,1))
        return sigmoid(result)

def avgModLoss(predicted, required):
    result = torch.abs(predicted - required)
    avg = Variable(mean(result,0), requires_grad=True)
    return avg

class CBOW:
    def __init__(self, window_size, dimension, noise):
        self.window_size = window_size
        self.vocab = set()
        self.word_indices = {}
        self.index_words = {}
        self.dimension = dimension
        self.vectors = []
        self.model = None
        self.noise = noise
        self.trainset = ()
        self.distribution = []
        self.rng = random.default_rng()

    def get_vocab(self, trainfile="../data/Electronics_5.json"):
        with open(trainfile, "r") as f:
            with open("tokenised.txt", "w") as tok:
                c = 0
                for line in f:
                    c += 1
                    words = word_tokenize(eval(line)["reviewText"])
                    words = [word.lower() for word in words]
                    tok.write(str(words) + '\n')
                    [self.vocab.add(word) for word in words]
                    if (c % 1000 == 0):
                        print("line number", c)
                    if (c == STOP_AT):
                        break

        self.vocab = list(self.vocab)

        with open("vocab.txt", "w") as f:
            for index, word in enumerate(self.vocab):
                self.word_indices.update({word: index})
                self.index_words.update({index: word})
                f.write(word + '\n')

    def create_model(self):
        self.model = nn.Sequential(
                        nn.Linear(len(self.vocab),self.dimension,bias=False),
                        MeanConcat(self.window_size*2),
                        GetSigmoid())

    def get_dataset(self):
        xTrain = []
        yTrain = []
        print("Getting distribution")
        c = 0
        with open("tokenised.txt", "r") as f:
            self.distribution = [0]*len(self.vocab)
            total_words = 0
            for line in f:
                c += 1
                words = eval(line)
                total_words += len(words)
                for word in words:
                    self.distribution[self.word_indices[word]] += 1

                if (c % 1000 == 0):
                    print("line number", c)
                if (c == STOP_AT):
                    break

            self.distribution = list(map(lambda x: x/total_words,
                                         self.distribution))

        print("Getting data")
        with open("tokenised.txt", "r") as f:
            #one_hots = F.one_hot(arange(len(self.vocab))).float()
            z = zeros(len(self.vocab))
            def one_hot(index):
                x = z.clone()
                x[index] = 1.
                return x

            c = 0
            for line in f:
                c += 1
                words = eval(line)

                window = words[:self.window_size*2+1]
                for t, word in enumerate \
                                 (words[self.window_size:-self.window_size]):
                    context = window.copy()
                    context.pop(self.window_size)
                    #context = words[t:t+self.window_size] \
                    #        + words[t+self.window_size+1:t+self.window_size*2+1]
                    try: window = window[1:] + [words[t + self.window_size*2+1]]
                    except: pass
                    xTrain.append(cat
                                    (list(map(lambda w: unsqueeze(one_hot(self.word_indices[w]),0),
                                               context +
                                               [word] +
                                               self.get_noise(context, word)))))
                    yTrain.append(tensor([1.] + [0.]*self.noise))

                #if (c % 1000 == 0):
                print("line number", c)
                if (c == STOP_AT):
                    break

        self.trainset = ([t.float() for t in xTrain], [t.float() for t in yTrain])

    def train_model(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(500):
            print("epoch", epoch)
            avg = 0
            for inp, outp in zip(self.trainset[0], self.trainset[1]):
                predicted = self.model(inp)
                loss = avgModLoss(predicted.flatten(), outp)
                avg += loss
                loss.backward()
                optimizer.step()

    def get_noise(self, context, word):
        #noise = []
        #for _ in range(self.noise):
        #    noise.append(random.choice(self.vocab))
        #noise = self.rng.choice(self.vocab, size=self.noise).tolist()
        rp = np.tile([1/len(self.vocab)]*len(self.vocab), (self.noise, 1))
        rs = np.random.random(rp.shape)
        rs /= rs.sum(axis=1)[:, np.newaxis]
        sp = rs - rp
        noise = np.argpartition(sp, 3, axis=1)[0, :3].tolist()
        noise = list(map(lambda i: self.vocab[i], noise))
        #print("noise is", noise)

        return noise
    
    def get_vectors(self):
        p = list(self.model.parameters())[0]
        p = transpose(p,0,1)
        self.vectors = [F.normalize(v,2.,0) for v in p]

        with open("cbow_vectors.txt", "w") as f:
            for word, vector in zip(self.vocab, self.vectors):
                f.write(word + '\n')
                f.write(str(vector.tolist()) + '\n')

    def get_top_ten(self, word):
        vec_words = []

        line1, line2 = (1, 1)
        with open("cbow_vectors.txt", "r") as f:
            for line1 in f:
                line1 = line1[:-1]
                line2 = f.readline()
                #line2 = re.sub(r' +', ',', line2).replace('[,', '[')
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
        print("top ten", list(map(lambda x: x[0], sort_v[:10])))


print("Initialising object")
letsgo = CBOW(2, 100, 3)
print("Getting vocab")
letsgo.get_vocab()
print("Creating model")
letsgo.create_model()
print("Getting dataset")
letsgo.get_dataset()
print("Training model")
letsgo.train_model()

with open('model_pkl', 'wb') as f:
    pickle.dump(letsgo, f)

print("Writing vectors to file")
letsgo.get_vectors()
