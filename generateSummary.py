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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set sum or mean for sentence embedding from w2v vectors
opr = 'wmd'
stop = 'withoutStop'
dataset = 'pasokh'

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

model = Word2Vec.load('models/w2v_300_combinedNormalized.bin')

for file in news_files: #200 News
	id = file[:-4]

	# if id != 'FAR.CU.13910203.004':
	# 	continue

	print(id)
	matrix = numpy.load('results/w2v_' + dataset + '_' + opr + '_' + stop + '/matrix/' + id + '.res.npy')
	# print(id, matrix.min())
	# print matrix

	# Force matrix to be symmetric
	matrix = (matrix + matrix.transpose())/2
	# print (matrix.transpose() == matrix).all()

	# Normal cosine similarity value between 0 and 1
	# matrix = matrix / 2 + 0.5
	print(matrix)

	num_cluster = 4
	if matrix.shape[0] < num_cluster + 1:
		num_cluster = matrix.shape[0] - 1

	cl = SpectralClustering(n_clusters=num_cluster,affinity='precomputed')

	print(num_cluster)
	try:
		sentence_clusters = cl.fit_predict(matrix)
	except:
		matrix = matrix / 2 + 0.5
		sentence_clusters = cl.fit_predict(matrix)
	print(sentence_clusters)

	# Load Title file
	with open('results/w2v_' + dataset + '_' + opr + '_' + stop + '/title_cosine/' + id + '.res') as f:
		title_similarity = f.read().splitlines()

	intermediate = numpy.empty((num_cluster, 0)).tolist()
	for index, sentence_id in enumerate(sentence_clusters):
		intermediate[sentence_id].append((index, title_similarity[index]))
	print(intermediate)

	title_intermediate = list()
	for small_list in intermediate:
		small_sorted = sorted(small_list, key=lambda sent: sent[1], reverse=True)
		title_intermediate.append(small_sorted)
	print(title_intermediate)

	# Select all sentences from One cluster
	# cluster_mean = list()
	# for i in range(0, num_cluster):
	# 	sum = 0
	# 	for j, d in enumerate(intermediate[i]):
	# 		print(d)
	# 		sum = sum + float(d[1])
	# 	print(sum, sum / (j + 1))
	# 	cluster_mean.append(sum / (j + 1))
	#
	# print(cluster_mean)
	# cluster_priority = list()
	# for value in sorted(cluster_mean):
	# 	for index, val in enumerate(cluster_mean):
	# 		if value == val:
	# 			cluster_priority.append(index)
	# 			break
	# print(cluster_priority)

	cluster_priority = list()
	ctr = Counter(sentence_clusters.ravel())
	cluster_priority.append(ctr.most_common(num_cluster))
	cluster_priority = [x[0] for x in cluster_priority[0]]
	print(cluster_priority)

	# Load Sentences file
	with open('results/w2v_' + dataset + '_' + opr + '_' + stop + '/sentences/'+id+'.txt') as f:
		sentences = f.read().splitlines()

	summary = ''
	sent_count = 0
	while(len(summary.split()) < 250 and sent_count < len(sentences)-1): #len(summary) < 250
		# cluster_no = cluster_priority[sent_count%num_cluster]
		# if(len(title_intermediate[cluster_no]) > 0):
		# 	sent_id = title_intermediate[cluster_no].pop(0)
		# 	summary = summary + sentences[sent_id[0]] + '\n'
		# sent_count += 1
		cluster_no = cluster_priority[sent_count % num_cluster]
		if (len(title_intermediate[cluster_no]) > 0):
			sent_id = title_intermediate[cluster_no].pop(0)
			summary = summary + sentences[sent_id[0]] + '\n'
		# sent_count += 1
		else:
			sent_count += 1

	filename = 'results/w2v_' + dataset + '_' + opr + '_' + stop + '/eval/files/system/'+id+'.sum'
	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	with open(filename, 'w') as g:
		g.write(summary)