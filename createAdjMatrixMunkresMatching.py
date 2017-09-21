from __future__ import division
import pandas
from hazm import *
from gensim.models import Doc2Vec, Word2Vec
import numpy
import math
from math import sqrt
import string

from os import system
import logging
import os
import sys
import pickle
from munkres import Munkres, print_matrix, make_cost_matrix
from scipy.stats import entropy
from numpy.linalg import norm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def square_rooted(x):
	return round(sqrt(sum([a * a for a in x])), 10)


def cosine_sim(x, y):
	numerator = sum(a * b for a, b in zip(x, y))
	denominator = square_rooted(x) * square_rooted(y)
	return round(numerator / float(denominator), 10)

def jaccard_similarity(x, y):
	intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
	union_cardinality = len(set.union(*[set(x), set(y)]))
	return intersection_cardinality / float(union_cardinality)

def JSD(P, Q):
	_P = P / norm(P, ord=1)
	_Q = Q / norm(Q, ord=1)
	_M = 0.5 * (_P + _Q)
	return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def dice_coefficient(a, b):
    """dice coefficient 2nt/na + nb."""
    a_bigrams = set(a)
    b_bigrams = set(b)
    overlap = len(a_bigrams & b_bigrams)
    return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))

def calculateSimilarity(sent1, sent2):
	# Case Correction
	sent1 = sent1.lower()
	sent2 = sent2.lower()

	# Tokenization
	tokens1 = word_tokenize(sent1)
	tokens2 = word_tokenize(sent2)

	# Remove punctuations
	tokens1 = [x for x in tokens1 if x not in string.punctuation]
	tokens2 = [x for x in tokens2 if x not in string.punctuation]

	# Remove Stopwords
	# stopWords = set(stopwords.words('english'))
	# tokens1 = [x for x in tokens1 if x not in stopWords]
	# tokens2 = [x for x in tokens2 if x not in stopWords]

	# Lemmatization
	# lemmatizer = WordNetLemmatizer()
	# tokens1 = [lemmatizer.lemmatize(x) for x in tokens1]
	# tokens2 = [lemmatizer.lemmatize(x) for x in tokens2]

	# Model has token?
	tokens1 = [x for x in tokens1 if x in model.vocab]
	tokens2 = [x for x in tokens2 if x in model.vocab]

	if len(tokens1) > 0 and len(tokens2) > 0:
		m = Munkres()

		pairMatrix = []
		for t1 in tokens1:
			tmpList = []
			for t2 in tokens2:
				tmpList.append(100 * JSD(model[t1], model[t2]))
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

		return total / len(indexes) / 100
		# return 2 * total / (len(tokens1) + len(tokens2)) / 100
	else:
		return 0

# Set sum or mean for sentence embedding from w2v vectors
opr = 'munkres_jsd'
stop = 'withoutStop'
dataset = 'pasokh'
trainedData = 'w2v_300_cleaned'

if opr == 'tfidf_mean':
	f = open('obj/words_tfidf.pkl', 'rb')
	tf_idf = pickle.load(f)
elif opr == 'tfidf_mean_STEM':
	f = open('obj/cleaned_stemmed/words_tfidf.pkl', 'rb')
	tf_idf = pickle.load(f)

if dataset == 'bistoon':
	orig_news_directory = '/home/mahmood/Desktop/Master/Thesis/Dataset/Biston/doted/'
	orig_title_directory = '/home/mahmood/Desktop/Master/Thesis/Dataset/Biston/title/'
elif dataset == 'pasokh':
	orig_news_directory = '/home/mahmood/Desktop/Master/Thesis/Dataset/Pasokh - Ijaz/Single -Dataset/Source/DUC/'
	orig_title_directory = '/home/mahmood/Desktop/Master/Thesis/Dataset/Pasokh - Ijaz/Single -Dataset/Source/Title/'
else:
	print('ERROR IN DATASET!')
	sys.exit()

# Read from seperated files Format in a directory
path = orig_news_directory
title_path = orig_title_directory
id_list = list()
text_list = list()
for filename in os.listdir(path):
	id_list.append(filename[:-4])
	with open(path+'/'+filename) as f:
		text_list.append(f.read().replace('\n', ''))

if trainedData == 'w2v_300_normalized':
	model = Word2Vec.load('models/w2v_300_combinedNormalized.bin')
elif trainedData == 'w2v_100_cleaned':
	model = Word2Vec.load('models/w2v_100_combinedNormalized_cleaned.bin')
elif trainedData == 'w2v_300_cleaned':
	model = Word2Vec.load('models/w2v_300_combinedNormalized_cleaned.bin')
elif trainedData == 'w2v_300_stemmed':
	model = Word2Vec.load('models/w2v_300_combinedNormalized_cleaned_stemmed.bin')

normalizer = Normalizer()

stopList = list()
with open('datasets/persian_stopwords.txt') as f:
	stopList = f.read().split('\n')

for index, t_list in enumerate(text_list):
	if index >= 0 and index < 100 :
		print(id_list[index])

		# if os.path.isfile('/home/mahmood/PycharmProjects/Word2Vec/results/w2v_300_cleaned/w2v_pasokh_munkres_jaccard_withoutStop/title_cosine' + id_list[index] != "HAM.CU.13901206.082":
		# 	continue
		if id_list[index] == "HAM.CU.13910107.090":
			continue
		elif id_list[index] == "FAR.EC.13910127.012":
			continue
		elif id_list[index] == "FAR.EC.13910203.008":
			pass
		elif id_list[index] == "JAM.SO.13910204.062":
			pass
		elif id_list[index] == "JAM.SC.13910204.002":
			pass
		elif id_list[index] == "TAB.CU.13900209.074":
			pass
		else:
			continue

		# Normalize News
		normalized_text = normalizer.normalize(text_list[index])

		# Extract Sentences
		sentences = sent_tokenize(normalized_text)
		reduced_sentences = []

		if stop == 'withStop':
			# Remove Punctuations
			for sentence in sentences:
				reduced_sentences.append(' '.join(
					word for word in word_tokenize(sentence) if word not in string.punctuation))

		elif stop == 'withoutStop':
			# Remove Stopwords and Punctuations
			for sentence in sentences:
				reduced_sentences.append(' '.join(
					word for word in word_tokenize(sentence) if word not in stopList and word not in string.punctuation))

		filename = 'results/' + trainedData + '/w2v_' + dataset + '_' + opr + '_' + stop + '/sentences/' + id_list[
			index] + '.txt'
		os.makedirs(os.path.dirname(filename), exist_ok=True)

		with open(filename, 'w') as f:
			for sent in sentences:
				f.write(sent)
				f.write('\n')

		# Title Operations
		with open(title_path + id_list[index] + '.txt') as t:
			title = t.read().replace('\n', ' ').replace('\r', '')
			pre_title = ' '.join(
				word for word in word_tokenize(title) if word not in stopList and word not in string.punctuation)

		# Save Titles Vector
		filename = 'results/' + trainedData + '/w2v_' + dataset + '_' + opr + '_' + stop + '/title_cosine/' + id_list[
			index] + '.res'
		os.makedirs(os.path.dirname(filename), exist_ok=True)

		with open(filename, 'w') as f:
			for sentence in reduced_sentences:
				val = calculateSimilarity(pre_title, sentence)
				f.write(str(val))
				f.write('\n')

		# Adjacency Matrix for Graph definition
		adj_matrix = numpy.zeros([len(sentences), len(sentences)])

		for i, sent in enumerate(reduced_sentences):
			for j, sent2 in enumerate(reduced_sentences):
				adj_matrix[i, j] = calculateSimilarity(sent, sent2)

		# print(adj_matrix)

		# Save Adjacency Matrix in results directory
		filename = 'results/' + trainedData + '/w2v_' + dataset + '_' + opr + '_' + stop + '/matrix/' + id_list[
			index] + '.res'
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		numpy.save(filename, adj_matrix)
