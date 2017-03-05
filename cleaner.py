import os, gensim, logging
from hazm import *
import sqlite3
import time
import string

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
	def __init__(self, filename):
		self.filename = filename
		conn = sqlite3.connect('/home/mahmood/Desktop/Master/Thesis/Dataset/dictionary/lab.sqlite')
		self.c = conn.cursor()
		# self.c.execute('SELECT Text FROM Words')
		# self.wordsList = []
		# for tuple in self.c.fetchall():
		# 	self.wordsList.append(tuple[0])

	def isValidWord(self, word):
		results = self.c.execute('SELECT * FROM Words WHERE Text=?', (word,))
		try:
			next(results)
			return True
		except StopIteration:
			return False

	def getWordFreq(self, word):
		results = self.c.execute('SELECT Frequency FROM Words WHERE Text=?', (word,))
		try:
			return int(results.fetchone()[0])
		except StopIteration:
			return 0

	def view(self):
		with open('datasets/cleaned.txt', 'w') as f:
			start_time = time.time()
			normalizer = Normalizer()
			cnt = [0, 0, 0, 0, 0, 0, 0]
			for line in open(self.filename):
				line = normalizer.normalize(line)
				cnt[0] += 1
				for w in word_tokenize(line):
					if self.isValidWord(w):
						cnt[1] += 1
						f.write(w + ' ')
					elif w in string.punctuation:
						cnt[2] += 1
						pass # Deleted
					elif w[-1:] in string.punctuation:
						if self.isValidWord(w[:-1]):
							cnt[3] += 1
							f.write(w[:-1] + ' ') # last character deleted
						else:
							cnt[4] += 1
							f.write(w + ' ')
							#TODO maybe word is concat of many punctuations
					else:
						lastFreqSum = 0
						w1, w2 = '', ''
						for i in range(1, len(w)):
							if self.isValidWord(w[:-i]) and self.isValidWord(w[-i:]):
								if self.getWordFreq(w[:-i]) + self.getWordFreq(w[-i:]) > lastFreqSum:
									w1, w2 = w[:-i], w[-i:]
									lastFreqSum = self.getWordFreq(w[:-i]) + self.getWordFreq(w[-i:])
						if w1 != '':
							cnt[5] += 1
							f.write(w1 + ' ' + w2 + ' ') # Replaced
						else:
							cnt[6] += 1
							f.write(w + ' ')
				f.write('\n')

			print("--- %s seconds ---" % (time.time() - start_time))
			print(cnt)

sentences = MySentences('/home/mahmood/PycharmProjects/Word2Vec/combined_sentences_normalized.txt').view()  # a memory-friendly iterator
