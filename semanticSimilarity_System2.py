from munkres import Munkres, print_matrix, make_cost_matrix
import sys
from scipy import stats
from os import listdir
from os.path import isfile, join
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
import logging
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def square_rooted(x):
	return round(sqrt(sum([a * a for a in x])), 10)

def cosine_sim(x, y):
	numerator = sum(a * b for a, b in zip(x, y))
	denominator = square_rooted(x) * square_rooted(y)
	return round(numerator / float(denominator), 10)

model = Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin.gz', binary=True)
# model = Word2Vec.load('models/w2v_100_combinedNormalized_cleaned.bin')

files = ['/home/mahmood/Desktop/Master/Thesis/Dataset/STS/2016/sts2016-english-with-gs-v1.0/STS2016.input.answer-answer.txt',
         '/home/mahmood/Desktop/Master/Thesis/Dataset/STS/2016/sts2016-english-with-gs-v1.0/STS2016.input.headlines.txt',
         '/home/mahmood/Desktop/Master/Thesis/Dataset/STS/2016/sts2016-english-with-gs-v1.0/STS2016.input.plagiarism.txt',
         '/home/mahmood/Desktop/Master/Thesis/Dataset/STS/2016/sts2016-english-with-gs-v1.0/STS2016.input.postediting.txt',
         '/home/mahmood/Desktop/Master/Thesis/Dataset/STS/2016/sts2016-english-with-gs-v1.0/STS2016.input.question-question.txt']

for file in files:
	with open(file) as f:
		with open('results/STS/SYSTEM_' + os.path.basename(file), 'w') as g:
			for line in f:
				[sent1, sent2] = line.split('\t')[0:2]

				# Case Correction
				sent1 = sent1.lower()
				sent2 = sent2.lower()

				# Tokenize and POS Tagging
				posTokens1 = pos_tag(word_tokenize(sent1))
				posTokens2 = pos_tag(word_tokenize(sent2))

				# posTokens Remove punctuations
				posTokens1 = [x for x in posTokens1 if x[0] not in string.punctuation]
				posTokens2 = [x for x in posTokens2 if x[0] not in string.punctuation]

				# Model has token?
				posTokens1 = [x for x in posTokens1 if x[0] in model.vocab]
				posTokens2 = [x for x in posTokens2 if x[0] in model.vocab]

				m = Munkres()

				pairMatrix = []
				for t1 in tokens1:
					tmpList = []
					for t2 in tokens2:
						tmpList.append(100 * cosine_sim(model[t1], model[t2]))
					pairMatrix.append(tmpList)

				cost_matrix = make_cost_matrix(pairMatrix, lambda cost: 100 - cost)
				indexes = m.compute(cost_matrix)
				# print_matrix(pairMatrix, msg='Lowest cost through this matrix:')
				total = 0
				for row, column in indexes:
					value = pairMatrix[row][column]
					total += value
					# print('(%d, %d) -> %d' % (row, column, value))
				# print('total cost: %d' % total)
				# print(total / len(indexes))
				g.write(str(int(total / len(indexes) / 20)))
				g.write('\n')

# print(stats.pearsonr([1, 2, 3, 4, 5], [1, 2, 3, 4, 4000]))
