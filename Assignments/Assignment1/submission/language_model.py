import sys
import random
import itertools
import math
import os.path
from collections  import defaultdict
from clean_tweets import tokenize

def get_testset(filepath, trainfile_name, testfile_name):
  """
  Splits a list (dataset) into
  training and testing data
  """
  corpus_file = open(filepath, 'r')
  lines = []                                    # Collects list of lines
  for line in corpus_file:                      # (strings) in the corpus
    lines.append(line)

  random.shuffle(lines)                         # Randomly divides corpus
  testset_strings = lines[:1000]                # into training
  trainset_strings = lines[1000:]               # and testing data

  corpus_file.close()

  trainset_file = open(trainfile_name, 'w')     # Stores training data
  for line in trainset_strings:                 # in the trainfile
    trainset_file.write(line)
  trainset_file.close()

  testset_file = open(testfile_name, 'w')       # and testing data
  for line in testset_strings:                  # in the testfile
    testset_file.write(line)
  testset_file.close()

def get_frequencies(trainfile_name, n):
  """
  Returns a list of dictionaries with the
  frequencies of each i-gram for i from
  1 to n, along with vocabulary
  """
  
  subgram_frequencies = [defaultdict(lambda: 0) for _ in range(n)]
                                                # List of dictionaries initialised

  trainset_file = open(trainfile_name, 'r')
  sentences = []
  freq_uni = defaultdict(lambda: 0)
  vocabulary = []
  for line in trainset_file:                    # First pass finds the frequencies
    sentence = tokenize(line)                   # of individual words
    sentences += [sentence]
    for word in sentence:
      freq_uni[word] += 1
    vocabulary += sentence
  trainset_file.close()
  
  vocabulary = list(set(vocabulary))
  voc_copy = vocabulary.copy()
  for i, word in enumerate(vocabulary):         # Edit the vocabulary
    if (freq_uni[word] <= 5):
      voc_copy.remove(word)
  vocabulary = set(voc_copy)
  vocabulary.add('<UNK>')
  vocabulary.add('<s>')
  vocabulary.add('</s>')
  
  cleaned_sentences = []
  for sentence in sentences:                    # Second pass
    cleaned = []
    for word in sentence:                       # replaces low-frequency words
      if (freq_uni[word] <= 5):                 # with <UNK>,
        cleaned += ['<UNK>']
      else:
        cleaned += [word]
    cleaned_sentences += [(['<s>']*(n-1)) + cleaned + (['</s>']*(n-1))]
                                                # pads with <s> and </s>,

  for sentence in cleaned_sentences:
    for index in range(n-1, len(sentence)):     # and creates the dictionaries
      ngram = (sentence[index],)
      for nvalue in range(1,n+1):
        subgram_frequencies[nvalue-1][ngram] += 1
        ngram = (sentence[index-nvalue],) + ngram

  return subgram_frequencies, vocabulary

def smooth(method, trainfile_name, n):
  """
  Returns a probability function
  """
  subgram_frequencies, vocabulary = get_frequencies(trainfile_name, n)
  subgram_items = [frequencies.items() for frequencies in subgram_frequencies]

  if (method == 'w'):                           # For Witten-Bell
    def prob(word, history, prob_dict):         # prob_dict stores all probabilities
      """                                       # calculated so far
      Takes in a word and its immediate
      history (n previous words) and
      returns the conditional probability
      """
      if (word not in vocabulary):              # Replace with <UNK>
        word = '<UNK>'
      for i in range(len(history)):
        if (history[i] not in vocabulary):
          history[i] = '<UNK>'

      try: p = prob_dict[tuple(history+[word])] # See if it has been calculated before
      except KeyError:                          # otherwise find p
        prev = history[-1]                      # base case n=2
        count = subgram_frequencies[1][(prev, word)]
        count_list = [freq for (w1, _), freq in subgram_items[1]
                           if (w1 == prev) and (freq > 0)]
        nw = sum(count_list)
        tw = len(count_list)
        if (tw == 0):
          tw = 2

        if (count == 0):
          zw = len(vocabulary) - tw
          if (zw == 0):
            zw = 2
          try: p = (tw / (zw * (nw + tw)))
          except: p = float("NaN")
        else:
          try: p = (count / (nw + tw))
          except: p = float("NaN")

        for i in range(3, n+1):                 # iterates from 3 to n
          ngram_frequencies = subgram_frequencies[i-1]
          ngram_frequencies_items = subgram_items[i-1]
          subhistory = history[-i+1:]
          count = ngram_frequencies[tuple(history[-i+1:] + [word])]
          count_list = [freq for context, freq in ngram_frequencies_items
                             if (list(context)[:-1] == subhistory) and (freq > 0)]
          total_hist_freq = sum(count_list)
          nw = len(count_list)
          if (nw == 0):
            nw = 2
          try: p = ((count + nw*p)/(total_hist_freq + nw))
          except: p = float("NaN")

        try: p = math.log(p)
        except: p = float("NaN")

      prob_dict[tuple(history+[word])] = p      # Store in dictionary
      return p

  elif (method == 'k'):                         # For Kneser-Ney 
    discount = 1

    def prob(word, history, prob_dict):         # prob_dict stores all probabilities
      """                                       # calculated so far
      As above
      """
      if (word not in vocabulary):
        word = '<UNK>'
      for i in range(len(history)):
        if (history[i] not in vocabulary):
          history[i] = '<UNK>'

      #print("Called for word", word, "history", history, "n", n)
      try: p = prob_dict[tuple(history+[word])] # See if it has been calculated before
      except KeyError:                          # otherwise find p
        pairs = [context for context, freq in subgram_items[1]
                         if (freq > 0)]
        numerator = sum(1 for _, w in pairs
                          if (w == word))
        denominator = len(pairs)
        if (denominator == 0):                  # Replace zero-count with a small number
          denominator = 2
        try: p = numerator/denominator
        except: return float("NaN")

        for i in range(2, n+1):                 # iterate from 2 to n
          ngram_frequencies = subgram_frequencies[i-1]
          ngram_frequencies_items = subgram_items[i-1]
          subhistory = history[-i+1:]
          count = ngram_frequencies[tuple(subhistory + [word])]
          count_list = [freq for context, freq in ngram_frequencies_items
                            if (list(context)[:-1] == subhistory) and (freq > 0)]
          total_count = sum(count_list)
          if (total_count == 0):
            total_count = 2
          nw = len(count_list)
          if (nw == 0):
            nw = 2
          try: p = (max(count - discount, 0)/total_count) + \
                   (discount/total_count) * nw * p
          except: p = float("NaN")

        #try:
        p = math.log(p)
        #except: p = float("NaN")
        prob_dict[tuple(history+[word])] = p
      return p

  return prob

def language_model(nvalue, smooth_method, corpus_path, trainfile_name, testfile_name):
  """
  Creates a language model, i.e.,
  a function from strings to
  their logprobabilities
  """
  if (not(os.path.exists(trainfile_name)) or not(os.path.exists(testfile_name))):
    print("Generating test and trainsets")
    get_testset(corpus_path, trainfile_name, testfile_name)

  logprobabilities = smooth(smooth_method, trainfile_name, nvalue)

  probs_dict = {}
  def lm(sentence, isTokenized='not'):
    """
    Takes a string, tokenises it,
    returns its probability according
    to training set & smoothing method
    """
    if (isTokenized == 'tok'):
      sentence = ['<s>']*(nvalue-1) + sentence + ['</s>']*(nvalue-1)
    else:
      sentence = ['<s>']*(nvalue-1) + tokenize(sentence) + ['</s>']*(nvalue-1)

    logprob = 0
    for index in range(nvalue-1, len(sentence)):
      lp = logprobabilities(sentence[index],
                            sentence[index-nvalue+1:index],
                            probs_dict)
      #print("Logprob of", sentence[index-nvalue+1:index+1], "is", lp)
      logprob += lp
    return logprob

  return lm

def perplexity(sentence, lang_mod, isTokenized='not'):
  """
  Takes a language model and a sentence
  and evaluates the perplexity of the
  model on the sentence
  """
  logprobability = lang_mod(sentence, isTokenized)
  if (isTokenized == 'tok'):
    n = len(sentence)
  else:
    n = len(tokenize(sentence))
  try: perp = math.exp(logprobability * (-1/n))
  except: perp = float("NaN")
  return perp

# Driver
n = int(sys.argv[1])
smoothing = sys.argv[2]
corpus = sys.argv[3]

lang_mod = language_model(n, smoothing, corpus, "train.txt", "test.txt")
input_sent = input("input sentence: ")
print(math.exp(lang_mod(input_sent)))

def generate_files():
  def generate_file(filename, dataset_file, lang_mod):
    f = open(filename, 'w')
    d = open(dataset_file, 'r')

    avg = 0
    i = 1
    not_nan = 0
    for sentence in d:
      if sentence[-1] == '\n':
        sentence = sentence[:-1]
      tokens = tokenize(sentence)
      perp = perplexity(tokens, lang_mod, 'tok')
      if (i % 100 == 0):
        print(i)
      if (perp != "nan"):
        avg += perp
        not_nan += 1
      f.write(sentence + '\t' + str(perp) + '\n')
      #print(sentence)
      #print(perp)
      i += 1
      if (i > 1000):
        break

    avg /= not_nan #len(dataset)
    f.seek(0, 0)
    f.write(str(avg) + '\n')
    f.close()
    d.close()

  lang_mod_1 = language_model(4, 'k', "../corpora/europarl-corpus.txt", "euro-train.txt", "euro-test.txt")
  print("Language model 1 generated.")
  generate_file("2020114001_LM1_train-perplexity.txt", "euro-train.txt", lang_mod_1)
  print("Training perplexities found.")
  generate_file("2020114001_LM1_test-perplexity.txt", "euro-test.txt", lang_mod_1)
  print("Testing perplexities found.")

  lang_mod_2 = language_model(4, 'w', "../corpora/europarl-corpus.txt", "euro-train.txt", "euro-test.txt")
  print("Language model 2 generated.")
  generate_file("2020114001_LM2_train-perplexity.txt", "euro-train.txt", lang_mod_2)
  print("Training perplexities found.")
  generate_file("2020114001_LM2_test-perplexity.txt", "euro-test.txt", lang_mod_2)
  print("Testing perplexities found.")

  lang_mod_3 = language_model(4, 'k', "../corpora/medical-corpus.txt", "medi-train.txt", "medi-test.txt")
  print("Language model 3 generated.")
  generate_file("2020114001_LM3_train-perplexity.txt", "medi-train.txt", lang_mod_3)
  print("Training perplexities found.")
  generate_file("2020114001_LM3_test-perplexity.txt", "medi-test.txt", lang_mod_3)
  print("Testing perplexities found.")

  lang_mod_4 = language_model(4, 'w', "../corpora/medical-corpus.txt", "medi-train.txt", "medi-test.txt")
  print("Language model 4 generated.")
  generate_file("2020114001_LM4_train-perplexity.txt", "medi-train.txt", lang_mod_4)
  print("Training perplexities found.")
  generate_file("2020114001_LM4_test-perplexity.txt", "medi-test.txt", lang_mod_4)
  print("Testing perplexities found.")
