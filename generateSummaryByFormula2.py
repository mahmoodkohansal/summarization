import numpy
from os import listdir
from os.path import isfile, join
from hazm import *
from collections import Counter
import logging
import os
import sys
from gensim.models import Doc2Vec, Word2Vec
import operator
import networkx as nx
from scipy import sparse
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set sum or mean for sentence embedding from w2v vectors
opr = 'tfidf_mean_STEM'
stop = 'withoutStop'
dataset = 'pasokh'
trainedData = 'w2v_300_stemmed'

if dataset == 'bistoon':
	orig_news_directory = '/home/mahmood/Desktop/Master/Thesis/Dataset/Biston/doted/'
	orig_title_directory = '/home/mahmood/Desktop/Master/Thesis/Dataset/Biston/title/'
elif dataset == 'pasokh':
	orig_news_directory = '/home/mahmood/Desktop/Master/Thesis/Dataset/Pasokh - Ijaz/Single -Dataset/Source/DUC/'
	orig_title_directory = '/home/mahmood/Desktop/Master/Thesis/Dataset/Pasokh - Ijaz/Single -Dataset/Source/Title/'
else:
	print('ERROR IN DATASET!')
	sys.exit()

orig_news_path = orig_news_directory
news_files = [f for f in listdir(orig_news_path) if isfile(join(orig_news_path, f))]

orig_title_path = orig_title_directory

if trainedData == 'w2v_300_normalized':
	model = Word2Vec.load('models/w2v_300_combinedNormalized.bin')
elif trainedData == 'w2v_100_cleaned':
	model = Word2Vec.load('models/w2v_100_combinedNormalized_cleaned.bin')
elif trainedData == 'w2v_300_stemmed':
	model = Word2Vec.load('models/w2v_300_combinedNormalized_cleaned_stemmed.bin')

for file in news_files: #200 News
	id = file[:-4]

	# if id != 'FAR.SP.13910202.013':
	# 	continue

	print(id)
	matrix = numpy.load('results/' + trainedData + '/w2v_' + dataset + '_' + opr + '_' + stop + '/matrix/'+id+'.res.npy')
	# print(id, matrix.min())
	# print matrix

	if opr == 'wmd':
		# Remove INF values in Matrix
		matrix[matrix == numpy.inf] = -numpy.inf
		matrix[matrix == -numpy.inf] = 2 * matrix.max()

		# Distance matrix to Similarity matrix
		beta = 0.1
		matrix = numpy.exp(-beta * matrix / matrix.std())

	# Force matrix to be symmetric
	matrix = (matrix + matrix.transpose())/2
	# print (matrix.transpose() == matrix).all()

	# Load Sentences file
	with open('results/' + trainedData + '/w2v_' + dataset + '_' + opr + '_' + stop + '/sentences/'+id+'.txt') as f:
		sentences = f.read().splitlines()

	similarity_graph = sparse.csr_matrix(matrix)
	nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
	try:
		scores = nx.pagerank(nx_graph, max_iter=1500)
	except:
		continue

	sentencesProperties = []

	# print(scores)
	# print(len(scores), len(sentences))
	ranked = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
	# for index, sent in enumerate(ranked):
		# print(sent[0])
		# print(sent[1])
		# print(sent[2])
		# print('--')

	for index, sent in enumerate(sentences):
		mainObject = dict()
		mainObject['index'] = index
		mainObject['sentence'] = sent
		mainObject['textrank_rank'] = scores[index]
		mainObject['final_score'] = mainObject['textrank_rank']
		sentencesProperties.append(mainObject)

	k = 5

	for index in range(0, len(sentences)):
		tmp = matrix[index][:]

		sentencesProperties[index]['similarity_index_list'] = numpy.argsort(tmp)[::-1][1:]
		sentencesProperties[index]['similarity_value_list'] = [tmp[x] for x in numpy.argsort(tmp)[::-1]][1:]

	for index in range(0, len(sentences)):
		up = numpy.mean(sentencesProperties[index]['similarity_value_list'][0:k])
		for neighbor_index in sentencesProperties[index]['similarity_index_list'][0:k]:
			down = numpy.mean(sentencesProperties[neighbor_index]['similarity_value_list'][0:k])
		sentencesProperties[index]['local_density_rank'] = up / down
		sentencesProperties[index]['final_score'] = sentencesProperties[index]['local_density_rank']

	usedSentences = list()

	summary = ''
	sent_count = 0
	while(len(summary.split()) < 250 and sent_count < len(sentences)-1): #len(summary) < 250
		for index in range(0, len(sentences)):
			if len(tuple(usedSentences)) > 0:
				tmp = matrix[index, tuple(usedSentences)]
				# print('==')
				# print(tmp)
				sentencesProperties[index]['most_similar_sentence_rank'] = tmp[numpy.argsort(tmp)[::-1]][0]
				sentencesProperties[index]['final_score'] /= tmp[numpy.argsort(tmp)[::-1]][0]
				# print(sentencesProperties[index]['final_score'])
				# print(sentencesProperties[index]['most_similar_sentence_rank'])
				# print(sentencesProperties[index]['textrank_rank'])
				# print(sentencesProperties[index]['local_density_rank'])

		sortedSentences = sorted(sentencesProperties, key=lambda item: item['final_score'], reverse=True)
		sortedSentences[0]['final_score'] = 0
		summary = summary + sortedSentences[0]['sentence'] + '\n'
		usedSentences.append(sortedSentences[0]['index'])
		sent_count += 1

	filename = 'results/' + trainedData + '/w2v_' + dataset + '_' + opr + '_' + stop + '/eval/files/formula2_ldr/system/'+id+'.sum'
	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	with open(filename, 'w') as g:
		g.write(summary)