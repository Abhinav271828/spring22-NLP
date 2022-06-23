from tokenise import tokenize
import time
import math
import torch
from torch import nn

class DataSet:
  def __init__(self, train_file, batch_size):
    self.file = open(train_file, "r")
    self.batch_size = batch_size
    self.vocab = []
    self.prefix_seqs = []
    self.batches = []
    self.words_to_indices = {}
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def get_vocab(self):
    freqs = {}
    m = 0
    nline = 0
    for line in self.file:
      nline += 1
      tokens = tokenize(line)
      self.vocab += tokens
      if (m <= len(tokens)):
        m = len(tokens)

      for token in tokens:
        try:
          freqs[token] += 1
        except KeyError:
          freqs[token] = 1

      try:
        pfx = [tokens[0]]
        for token in tokens[1:]:
          pfx.append(token)
          self.prefix_seqs.append(pfx.copy())
      except IndexError:
        continue

      if (nline % 5000 == 0):
        print("Line number", nline)

    self.maxlength = m
    self.vocab = list(set(self.vocab))
    self.vocab = list(filter(lambda wd: freqs[wd] >= 5, self.vocab))
    self.vocab.append('<unk>')
    self.vocab.append('<pad>')
    print("Vocabulary created.")
  
  def index(self, word):
    try:
      res = self.words_to_indices[word]
    except KeyError:
      res = len(self.vocab) - 2
    return res

  def get_batches(self):
    self.get_vocab()

    self.words_to_indices = {word: index
                               for index, word in enumerate(self.vocab)}

    pad_index = len(self.vocab) - 1
    for i in range(len(self.prefix_seqs)):
      curr_seq = self.prefix_seqs[i]
      curr_seq = [self.index(w) for w in curr_seq]
      self.prefix_seqs[i] = [pad_index]*(self.maxlength-len(curr_seq)) \
                                + curr_seq

    self.prefix_seqs = torch.tensor(self.prefix_seqs, device=self.device)
    self.batches = torch.split(self.prefix_seqs, self.batch_size)

    self.context_word_batches = []
    for batch in self.batches:
      self.context_word_batches.append((batch[:,:-1], batch[:,-1]))

torch.set_printoptions(precision=20)

class LSTMModel(nn.Module):
  def __init__(self, repr_dim, hidden_size, learning_rate, momentum, epsilon,
                     trainset):
    super().__init__()
    self.repr_dim = repr_dim
    self.hidden_size = hidden_size
    self.model = None
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.trainset = trainset

    self.lr = learning_rate
    self.momentum = momentum
    self.epsilon = epsilon

    self.embedding_layer = nn.Embedding(len(trainset.vocab), self.repr_dim,
                                        device=self.device)
    self.lstm = nn.LSTM(input_size=self.repr_dim,
                        hidden_size=self.hidden_size,
                        batch_first=True)
    self.output = nn.Linear(self.hidden_size, len(trainset.vocab),
                            bias=False, device=self.device)
    self.to(self.device)

  def forward(self, contexts):
    embeddings = self.embedding_layer(contexts)

    hidden_reps, _ = self.lstm(embeddings)
    
    output = self.output(hidden_reps[:,-1])

    return output

def train_one_epoch(model, criterion, optimiser):
  avg_epoch_loss = 0
  for i, (contexts, words) in enumerate(TRAINSET.context_word_batches):
    optimiser.zero_grad()
    predicted = model.forward(contexts)
    target = words
    loss = criterion(predicted, target)
    avg_epoch_loss += loss.item()
    loss.backward()
    optimiser.step()

    if (i % 500 == 0):
      print(loss.item(), end=' | ')

  avg_epoch_loss /= len(TRAINSET.batches)
  return avg_epoch_loss
  
def train(model):
  criterion = nn.CrossEntropyLoss()
  optimiser = torch.optim.SGD(model.parameters(),
                              lr=model.lr, momentum=model.momentum)

  prev_loss = -math.inf
  for epoch in range(100):
    avg_epoch_loss = train_one_epoch(model, criterion, optimiser)

    print(avg_epoch_loss)

    if (abs(avg_epoch_loss - prev_loss) <= model.epsilon):
      break
    prev_loss = avg_epoch_loss

def get_perp(model, sentence):
  if (len(sentence) == 0):
    return float("NaN")
  contexts = sentence[:,:-1]
  words = sentence[:,-1]

  prob_dists = model.forward(contexts)

  sum_logprob = 0
  for i, word in enumerate(words):
    prob = prob_dists[i][model.trainset.index(word)]
    try: sum_logprob += math.log(prob)
    except: sum_logprob = -math.inf

  sum_logprob /= len(sentence[-1])

  return math.exp(sum_logprob)

def perp_file(model, filename, perp_filename):
  with open(filename, "r") as f:
    writefile = open(perp_filename, "w")
    avg_perp = 0
    n_lines = 0
    pad_index = len(model.trainset.vocab) - 1
    m = model.trainset.maxlength
    for line in f:
      tokens = tokenize(line)
      indices = [model.trainset.index(w) for w in tokens]
      sentence = [indices[:i] for i in range(2,len(indices))]
  
      sentence = [ ([pad_index]*(m-len(prefix)) + prefix)
                  for prefix in sentence]
      try: sentence = torch.tensor(sentence, device=model.device)
      except:
        writefile.write('\n')
        return 0

      perp = get_perp(model, sentence)

      writefile.write(line + '\t' + str(perp) + '\n')

      avg_perp += perp
      n_lines += 1

      if (n_lines % 1000 == 0):
        print("Line number", n_lines)
    
    avg_perp /= n_lines
    writefile.seek(0,0)
    writefile.write(str(avg_perp) + '\n')
    writefile.close()

  return avg_perp


TRAINSET = DataSet("train.europarl", 256)
TRAINSET.get_batches()

saved_model = sys.argv[1]
loaded_model = LSTMModel(100, 150, 0.001, 0.9, 0.001, None)
loaded_model.load_state_dict(torch.load(saved_model))

inp_sentence = input("input sentence: ")
tokens = tokenize(line)
indices = [TRAINSET.index(w) for w in tokens]
sentence = [indices[:i] for i in range(2,len(indices))]
sentence = [ ([pad_index]*(m-len(prefix)) + prefix)
            for prefix in sentence]

perp = get_perp(loaded_model, sentence)
print(str(perp) + '\n')

#lm = LSTMModel(100, 150, 0.001, 0.9, 0.001, TRAINSET)
#
#train(lm)
#
#torch.save(lm.state_dict(), "eng_lm.pth")
#
#perp_file(lm, "train.europarl", "train_perp.txt")
#perp_file(lm, "test.europarl", "test_perp.txt")

#TRAINSET = DataSet("train.news", 256)
#TRAINSET.get_batches()
#lm = LSTMModel(100, 150, 0.001, 0.9, 0.001, TRAINSET)
#train(lm)
#torch.save(lm.state_dict(), "fr_lm.pth")
