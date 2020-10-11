from __future__ import division
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import numpy
from sklearn.cluster import SpectralClustering
from os import listdir
from os.path import isfile, join
from hazm import *
from collections import Counter
import logging
import os
import sys
from gensim.models import Word2Vec
from gensim import corpora
from math import sqrt
import pickle
import string
from munkres import Munkres, print_matrix, make_cost_matrix

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

	if typ == 'mean':
		for word in sent_words:
			word = word.replace('\ufeff', '')
			try:
				sent_words_vectors.append(model[word])
			except:
				pass
		sent_vector = numpy.mean(sent_words_vectors, axis=0)

	elif typ == 'tfidf_mean':
		for word in sent_words:
			word = word.replace('\ufeff', '')
			try:
				sent_words_vectors.append(model[word] * tf_idf[word])
			except:
				pass
		sent_vector = numpy.mean(sent_words_vectors, axis=0)

	elif typ == 'tfidf_mean_STEM':
		for word in sent_words:
			word = word.replace('\ufeff', '')
			if word not in string.punctuation:
				try:
					sent_words_vectors.append(model[word] * tf_idf[word])
				except:
					pass
		sent_vector = numpy.mean(sent_words_vectors, axis=0)

	return sent_vector

def calculateSentencesScore(sentences, dictionary, count_dict, ignored_words):
	ign_ids = [dictionary.token2id[ign_wrd] for ign_wrd in ignored_words if ign_wrd in dictionary.token2id]
	dictionary.filter_tokens(ign_ids)

	sent_scores = []
	for line in sentences:
		tmp_score = 0
		tmp_count = 0
		for word in word_tokenize(line):
			try:
				tmp_score += count_dict[dictionary.token2id[word]][1]
				tmp_count += 1
			except:
				pass
		if tmp_count != 0:
			sent_scores.append(tmp_score / tmp_count)
		else:
			sent_scores.append(0)
	return sent_scores

def calculateIntermediate(num_cluster, sentence_clusters, sent_scores):
	intermediate = numpy.empty((num_cluster, 0)).tolist()
	for index, sentence_id in enumerate(sentence_clusters):
		intermediate[sentence_id].append((index, sent_scores[index]))
	# print(intermediate)

	intermediate2 = list()
	for small_list in intermediate:
		if small_list != 0:
			small_sorted = sorted(small_list, key=lambda sent: sent[1], reverse=True)
			intermediate2.append(small_sorted)
	print(intermediate2)
	return intermediate2

def calculateSimilarity(sent1, sent2, isTitle):
	# Case Correction
	sent1 = sent1.lower()
	sent2 = sent2.lower()

	# Tokenization
	normalizer = Normalizer()
	_tokens1 = word_tokenize(normalizer.normalize(sent1.replace('\u200f', '').replace('\ufeff', '')))
	_tokens2 = word_tokenize(normalizer.normalize(sent2.replace('\u200f', '').replace('\ufeff', '')))

	# Remove punctuations
	_tokens1 = [x for x in _tokens1 if x not in string.punctuation]
	_tokens2 = [x for x in _tokens2 if x not in string.punctuation]

	# Trigram Replacement
	tokens1_tmp = []
	i = 0
	while i < len(_tokens1) - 2:
		trigram = _tokens1[i] + '_' + _tokens1[i + 1] + '_' + _tokens1[i + 2]
		if trigram in model.wv.vocab:
			tokens1_tmp.append(_tokens1[i] + '_' + _tokens1[i + 1] + '_' + _tokens1[i + 2])
			i += 3
		else:
			tokens1_tmp.append(_tokens1[i])
			i += 1
	try:
		tokens1_tmp.append(_tokens1[len(_tokens1) - 2])
		tokens1_tmp.append(_tokens1[len(_tokens1) - 1])
	except:
		pass

	tokens2_tmp = []
	i = 0
	while i < len(_tokens2) - 2:
		trigram = _tokens2[i] + '_' + _tokens2[i + 1] + '_' + _tokens2[i + 2]
		if trigram in model.wv.vocab:
			tokens2_tmp.append(_tokens2[i] + '_' + _tokens2[i + 1] + '_' + _tokens2[i + 2])
			i += 3
		else:
			tokens2_tmp.append(_tokens2[i])
			i += 1
	try:
		tokens2_tmp.append(_tokens2[len(_tokens2) - 2])
		tokens2_tmp.append(_tokens2[len(_tokens2) - 1])
	except:
		pass

	# Bigrams Replacement
	tokens1 = []
	i = 0
	while i < len(tokens1_tmp) - 1:
		bigram = tokens1_tmp[i] + '_' + tokens1_tmp[i + 1]
		if bigram in model.wv.vocab:
			tokens1.append(tokens1_tmp[i] + '_' + tokens1_tmp[i + 1])
			i += 2
		else:
			tokens1.append(tokens1_tmp[i])
			i += 1
	try:
		tokens1.append(_tokens1[len(tokens1_tmp) - 1])
	except:
		pass

	tokens2 = []
	i = 0
	while i < len(tokens2_tmp) - 1:
		bigram = tokens2_tmp[i] + '_' + tokens2_tmp[i + 1]
		if bigram in model.wv.vocab:
			tokens2.append(tokens2_tmp[i] + '_' + tokens2_tmp[i + 1])
			i += 2
		else:
			tokens2.append(tokens2_tmp[i])
			i += 1
	try:
		tokens2.append(_tokens2[len(tokens2_tmp) - 1])
	except:
		pass

	# Model has token?
	tokens1 = [x for x in tokens1 if x in model.wv.vocab]
	tokens2 = [x for x in tokens2 if x in model.wv.vocab]

	tmp_matched_ngrams = 0
	tmp_matched_1grams = 0

	if len(tokens1) > 0 and len(tokens2) > 0:
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
			if not isTitle:
				if '_' in tokens1[row]:
					tmp_matched_ngrams += 1
				elif '_' in tokens2[column]:
					tmp_matched_ngrams += 1
				else:
					tmp_matched_1grams += 1
		# print('(%d, %d) -> %d' % (row, column, value))
		# print('total cost: %d' % total)
		# print(total / len(indexes))

		# return total / len(indexes) / 100
		return [2 * total / (len(tokens1) + len(tokens2)) / 100, tmp_matched_ngrams, tmp_matched_1grams]
	else:
		return [0, 0, 0]

def summarize_base(input_str, summ_len):
	summ_len = int(summ_len)
	opr = 'tfidf_mean'
	stop = 'withoutStop'

	if opr == 'tfidf_mean':
		f = open('resources/words_tfidf.pkl', 'rb')
		tf_idf = pickle.load(f)
	elif opr == 'tfidf_mean_STEM':
		f = open('obj/cleaned_stemmed/words_tfidf.pkl', 'rb')
		tf_idf = pickle.load(f)

	stopList = list()
	with open('resources/persian_stopwords.txt') as f:
		stopList = f.read().split('\n')

	normalizer = Normalizer()

	news_text = input_str
	dictionary = corpora.Dictionary(word_tokenize(normalizer.normalize(line)) for line in sent_tokenize(news_text))

	print(sent_tokenize(news_text))

	count_dict = dictionary.doc2bow(word_tokenize(normalizer.normalize(news_text)))

	# Extract Sentences
	normalized_text = normalizer.normalize(news_text)
	sentences = sent_tokenize(normalized_text)

	sent_scores = calculateSentencesScore(sentences, dictionary, count_dict, [])

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

	# Adjacency Matrix for Graph definition
	matrix = numpy.zeros([len(reduced_sentences), len(reduced_sentences)])

	for i, sent in enumerate(reduced_sentences):
		for j, sent2 in enumerate(reduced_sentences):
			vec1 = sent2vec(model, sent, opr)
			vec2 = sent2vec(model, sent2, opr)
			try:
				matrix[i, j] = cosine_sim(vec1, vec2)
			except:
				matrix[i, j] = numpy.float32(0)

	# Force matrix to be symmetric
	matrix = (matrix + matrix.transpose()) / 2

	# Normal cosine similarity value between 0 and 1
	# matrix = matrix / 2 + 0.5
	# print(matrix)
	# print('-----------------------------------------------------------')

	num_cluster = 4
	if matrix.shape[0] < num_cluster + 1:
		num_cluster = matrix.shape[0] - 1

	cl = SpectralClustering(n_clusters=num_cluster, affinity='precomputed')

	# print(num_cluster)
	try:
		sentence_clusters = cl.fit_predict(matrix)
	except:
		matrix = matrix / 2 + 0.5
		sentence_clusters = cl.fit_predict(matrix)
	# print(sentence_clusters)

	intermediate2 = calculateIntermediate(num_cluster, sentence_clusters, sent_scores)

	cluster_priority = list()
	ctr = Counter(sentence_clusters.ravel())
	cluster_priority.append(ctr.most_common(num_cluster))
	cluster_priority = [x[0] for x in cluster_priority[0]]
	print(cluster_priority)

	summary = ''
	sent_count = 0
	while (len(summary.split()) < summ_len and sent_count < len(sentences) - 1):
		print(len(summary.split()))
		print(sent_count)
		print(len(sentences) - 1)
		print('-----')

		cluster_no = cluster_priority[sent_count % num_cluster]
		if (len(intermediate2[cluster_no]) > 0):
			sent_id = intermediate2[cluster_no].pop(0)
			summary = summary + sentences[sent_id[0]] + '\n<br>'
			sent_count += 1
		# intermediate2 = updateScores(sent_id[0])
		else:
			sent_count += 1

		print(len(summary.split()))
		print(sent_count)
		print(len(sentences) - 1)
		print('============================')

	return [summary, len(input_str.split()), len(summary.split())]

def summarize_advanced(input_str, summ_len):
	summ_len = int(summ_len)
	opr = 'tfidf_mean'
	stop = 'withoutStop'

	if opr == 'tfidf_mean':
		f = open('resources/words_tfidf.pkl', 'rb')
		tf_idf = pickle.load(f)
	elif opr == 'tfidf_mean_STEM':
		f = open('obj/cleaned_stemmed/words_tfidf.pkl', 'rb')
		tf_idf = pickle.load(f)

	stopList = list()
	with open('resources/persian_stopwords.txt') as f:
		stopList = f.read().split('\n')

	normalizer = Normalizer()

	news_text = input_str
	dictionary = corpora.Dictionary(word_tokenize(normalizer.normalize(line)) for line in sent_tokenize(news_text))

	print(sent_tokenize(news_text))

	count_dict = dictionary.doc2bow(word_tokenize(normalizer.normalize(news_text)))

	# Extract Sentences
	normalized_text = normalizer.normalize(news_text)
	sentences = sent_tokenize(normalized_text)

	sent_scores = calculateSentencesScore(sentences, dictionary, count_dict, [])

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

	# Adjacency Matrix for Graph definition
	matrix = numpy.zeros([len(reduced_sentences), len(reduced_sentences)])

	for i, sent in enumerate(reduced_sentences):
		for j, sent2 in enumerate(reduced_sentences):
			tmp_res = calculateSimilarity(sent, sent2, isTitle=False)
			matrix[i, j] = tmp_res[0]

	# Force matrix to be symmetric
	matrix = (matrix + matrix.transpose()) / 2

	# Normal cosine similarity value between 0 and 1
	# matrix = matrix / 2 + 0.5
	# print(matrix)
	# print('-----------------------------------------------------------')

	num_cluster = 4
	if matrix.shape[0] < num_cluster + 1:
		num_cluster = matrix.shape[0] - 1

	cl = SpectralClustering(n_clusters=num_cluster, affinity='precomputed')

	# print(num_cluster)
	try:
		sentence_clusters = cl.fit_predict(matrix)
	except:
		matrix = matrix / 2 + 0.5
		sentence_clusters = cl.fit_predict(matrix)
	# print(sentence_clusters)

	intermediate2 = calculateIntermediate(num_cluster, sentence_clusters, sent_scores)

	cluster_priority = list()
	ctr = Counter(sentence_clusters.ravel())
	cluster_priority.append(ctr.most_common(num_cluster))
	cluster_priority = [x[0] for x in cluster_priority[0]]
	print(cluster_priority)

	summary = ''
	sent_count = 0
	while (len(summary.split()) < summ_len and sent_count < len(sentences) - 1):
		print(len(summary.split()))
		print(sent_count)
		print(len(sentences) - 1)
		print('-----')

		cluster_no = cluster_priority[sent_count % num_cluster]
		if (len(intermediate2[cluster_no]) > 0):
			sent_id = intermediate2[cluster_no].pop(0)
			summary = summary + sentences[sent_id[0]] + '\n<br>'
			sent_count += 1
		# intermediate2 = updateScores(sent_id[0])
		else:
			sent_count += 1

		print(len(summary.split()))
		print(sent_count)
		print(len(sentences) - 1)
		print('============================')

	return [summary, len(input_str.split()), len(summary.split())]

app = Flask(__name__) #create the Flask app
model = Word2Vec.load('resources/w2v_300cleaned_phrases+word2vec.bin')

@app.route('/summarize', methods=['POST'])
def query_example():
	if request.method == 'POST':
		if request.form.get('method') == 1:
			res = summarize_base(request.form.get('input'), request.form.get('len'))
		else:
			res = summarize_advanced(request.form.get('input'), request.form.get('len'))
		return jsonify({'summary': res[0], 'input_count': res[1], 'summary_count': res[2]})

if __name__ == '__main__':
	app.run(debug=True, port=5000) #run app in debug mode on port 5000
