import nltk
import csv
import string
from gensim.models import Phrases
from gensim.models import Word2Vec
from nltk.corpus import stopwords


model = Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin.gz', binary=True)

vocab = model.vocab.keys()

for i in range(0, 2000):
	print(vocab[i])