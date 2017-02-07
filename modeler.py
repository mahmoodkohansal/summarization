import os, gensim, logging
from hazm import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		normalizer = Normalizer()
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, fname)):
				line = normalizer.normalize(line)
				yield line.split()

sentences = MySentences('/home/mahmood/PycharmProjects/Word2Vec/sentences')  # a memory-friendly iterator
# print sentences

model = gensim.models.Word2Vec(sentences, size=300, window=5, workers=4)

model.save('models/w2v_300_combinedNormalized.bin')

