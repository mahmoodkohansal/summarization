import numpy
from sklearn.cluster import SpectralClustering
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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set sum or mean for sentence embedding from w2v vectors
opr = 'munkres'
stop = 'withoutStop'
dataset = 'pasokh'
trainedData = 'w2v_300_cleaned'

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
elif trainedData == 'w2v_300_cleaned':
	model = Word2Vec.load('models/w2v_300_combinedNormalized_cleaned.bin')
elif trainedData == 'w2v_300_stemmed':
	model = Word2Vec.load('models/w2v_300_combinedNormalized_cleaned_stemmed.bin')

for file in news_files: #200 News
	id = file[:-4]

	if id == 'FAR.EC.13910127.012':
		continue
	if id == 'HAM.CU.13910107.090':
		continue

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
	# print(scores)
	# print(len(scores), len(sentences))
	ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)),reverse=True)

	summary = ''
	sent_count = 0
	while(len(summary.split()) < 250 and sent_count < len(sentences)-1): #len(summary) < 250
		summary = summary + ranked[sent_count][1] + '\n'
		sent_count += 1

	filename = 'results/' + trainedData + '/w2v_' + dataset + '_' + opr + '_' + stop + '/eval/files/textrank_old/system/'+id+'.sum'
	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	with open(filename, 'w') as g:
		g.write(summary)
