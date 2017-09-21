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
from difflib import SequenceMatcher

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set sum or mean for sentence embedding from w2v vectors
opr = 'mean'
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

summariesDir = '/home/mahmood/Desktop/Master/Thesis/Dataset/Pasokh - Ijaz/Single -Dataset/Summ/Extractive/'

p = open('pasokh_stats.csv', 'w')
p.write('News File Name, Sentences, Sum1 Sents, Sum2 Sents, Sum3 Sents, Sum4 Sents, Sum5 Sents, Words, Sum1 Words, Sum2 Words, Sum3 Words, Sum4 Words, Sum5 Words')
p.write('\n')

# p = open('clustering_oracle.csv', 'w')
# p.write('News File Name, Sentences, Words, cl#1, cl#2, cl#3, cl#4, cl#1, cl#2, cl#3, cl#4, cl#1, cl#2, cl#3, cl#4, cl#1, cl#2, cl#3, cl#4, cl#1, cl#2, cl#3, cl#4')
# p.write('\n')

for file in news_files: #200 News
	id = file[:-4]

	# if id != 'TAB.PO.13910202.032':
	# 	continue

	print(id)
	matrix = numpy.load('results/' + trainedData + '/w2v_' + dataset + '_' + opr + '_' + stop + '/matrix/' + id + '.res.npy')

	# Force matrix to be symmetric
	matrix = (matrix + matrix.transpose())/2

	print(matrix)
	print('-----------------------------------------------------------')

	num_cluster = 4
	if matrix.shape[0] < num_cluster + 1:
		num_cluster = matrix.shape[0] - 1

	cl = SpectralClustering(n_clusters=num_cluster,affinity='precomputed')

	try:
		sentence_clusters = cl.fit_predict(matrix)
	except:
		matrix = matrix / 2 + 0.5
		sentence_clusters = cl.fit_predict(matrix)
	print(sentence_clusters)

	# Load Sentences file
	with open('results/' + trainedData + '/w2v_' + dataset + '_' + opr + '_' + stop + '/sentences/' + id + '.txt') as f:
		sentences = f.read().splitlines()

	newsWordsCount = 0
	for ss in sentences:
		newsWordsCount += len(ss.split())

	intermediate = []
	for index, sentence_id in enumerate(sentence_clusters):
		intermediate.append((sentence_id, index, sentences[index]))

	# for y in intermediate:
		# print(y[0]) # Cluster Number
		# print(y[1]) # Sentence Index
		# print(y[2]) # Sentence Text
		# print('---')

	sumSentenceCount = []
	sumWordsCount = []
	sumEachClusterCount = []
	for ffile in os.listdir(summariesDir):
		if ffile.startswith(id):
			# print(ffile)
			summarySentencesList = []
			with open(summariesDir + ffile) as f:
				txt = f.read().replace('\n', '')

				normalizer = Normalizer()
				normalized_text = normalizer.normalize(txt)
				summarySentencesList = sent_tokenize(normalized_text)

				sumSentenceCount.append(len(summarySentencesList))
				x = 0
				for dd in summarySentencesList:
					x += len(dd.split())
				sumWordsCount.append(x)

				tmp = [0] * 4
				for sumSent in summarySentencesList:
					print(sumSent)
					a = max([(SequenceMatcher(None, sumSent, origSent[2]).ratio(), origSent) for origSent in intermediate])
					print(a[0])
					tmp[a[1][0]] += 1

				sumEachClusterCount.append(tmp)

				print('++++++++============++++++++++++============+++++++++++==========')


	p.write(str(id) + ',' + str(len(intermediate)) + ',')
	for i in range(0, 5):
		try:
			p.write(str(sumSentenceCount[i]) + ',')
		except:
			p.write(',')
	p.write(str(newsWordsCount) + ',')
	for i in range(0, 5):
		try:
			p.write(str(sumWordsCount[i]) + ',')
		except:
			p.write(',')
	p.write('\n')

	# p.write(str(id) + ',' + str(len(intermediate)) + ',' + str(newsWordsCount) + ',')
	# for i in range(0, 5):
	# 	for j in range(0, 4):
	# 		try:
	# 			p.write(str(sumEachClusterCount[i][j]) + ',')
	# 		except:
	# 			p.write(',')
	# p.write('\n')