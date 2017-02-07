from __future__ import division
import pandas
from hazm import *
from gensim.models import Doc2Vec, Word2Vec
import numpy
import math
from math import sqrt
import string

from os import system
from sklearn.cluster import SpectralClustering
import logging
import os
import sys
import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def square_rooted(x):
	return round(sqrt(sum([a * a for a in x])), 10)


def cosine_sim(x, y):
	numerator = sum(a * b for a, b in zip(x, y))
	denominator = square_rooted(x) * square_rooted(y)
	return round(numerator / float(denominator), 10)

def sent2vec(model, sent, typ):
	normalizer = Normalizer()
	sentence = normalizer.normalize(sent)
	sent_words = word_tokenize(sentence)
	sent_words_vectors = list()
	for word in sent_words:
		word = word.replace('\ufeff', '')
		try:
			sent_words_vectors.append(model[word] * tf_idf[word])
		except:
			pass

	if typ == 'sum':
		sent_vector = numpy.sum(sent_words_vectors, axis=0)
	else:
		sent_vector = numpy.mean(sent_words_vectors, axis=0)
	return sent_vector

# Set sum or mean for sentence embedding from w2v vectors
opr = 'tfidf_mean'
stop = 'withoutStop'
dataset = 'pasokh'

f = open('obj/words_tfidf.pkl', 'rb')
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

model = Word2Vec.load('models/w2v_300_combinedNormalized.bin')

normalizer = Normalizer()

stopList = list()
with open('datasets/persian_stopwords.txt') as f:
	stopList = f.read().split('\n')

for index, t_list in enumerate(text_list):
	print(id_list[index])

	# Normalize News
	normalized_text = normalizer.normalize(text_list[index])

	# Extract Sentences
	sentences = sent_tokenize(normalized_text)
	reduced_sentences = []

	# Remove Stopwords and Punctuations
	for sentence in sentences:
		reduced_sentences.append(' '.join(word for word in word_tokenize(sentence) if word not in stopList and word not in string.punctuation))

	filename = 'results/test__w2v_' + dataset + '_' + opr + '_' + stop + '/sentences/' + id_list[index] + '.txt'
	os.makedirs(os.path.dirname(filename), exist_ok=True)

	with open(filename, 'w') as f:
		for sent in sentences:
			f.write(sent)
			f.write('\n')

	# Title Operations
	with open(title_path + id_list[index] + '.txt') as t:
		title = t.read().replace('\n', ' ').replace('\r', '')
		pre_title = ' '.join(word for word in word_tokenize(title) if word not in stopList and word not in string.punctuation)
		title_vector = sent2vec(model, pre_title, opr)

	# print(title_vector)
	# print(len(sentences))

	# Save Titles Vector
	filename = 'results/test__w2v_' + dataset + '_' + opr + '_' + stop + '/title_cosine/' + id_list[index] + '.res'
	os.makedirs(os.path.dirname(filename), exist_ok=True)

	with open(filename, 'w') as f:
		for sentence in reduced_sentences:
			sent_vec = sent2vec(model, sentence, opr)
			print(sent_vec.size)
			if(sent_vec.size != 1): # sentence is not short
				try:
					cosine_val = cosine_sim(title_vector, sent_vec)
				except:
					pass
			else:
				cosine_val = numpy.float32(0)
			f.write(cosine_val.astype('str'))
			f.write('\n')
			# print(sent)

	# Adjacency Matrix for Graph definition
	adj_matrix = numpy.zeros([len(sentences), len(sentences)])

	for i, sent in enumerate(reduced_sentences):
		for j, sent2 in enumerate(reduced_sentences):
			vec1 = sent2vec(model, sent, opr)
			vec2 = sent2vec(model, sent2, opr)
			try:
				adj_matrix[i, j] = cosine_sim(vec1, vec2)
			except:
				adj_matrix[i, j] = numpy.float32(0)

	print(adj_matrix)

	# Save Adjacency Matrix in results directory
	filename = 'results/test__w2v_' + dataset + '_' + opr + '_' + stop + '/matrix/' + id_list[index] + '.res'
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	numpy.save(filename, adj_matrix)
