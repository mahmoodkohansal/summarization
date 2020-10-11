import os, gensim, logging
from hazm import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Sentences(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		normalizer = Normalizer()
		for file_name in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, file_name)):
				line = normalizer.normalize(line)
				yield word_tokenize(line)

sentences = Sentences('Word2Vec/sentences')

model = gensim.models.Word2Vec(sentences, size=300, window=5, workers=8)

model.save('models/w2v_300_model.bin')