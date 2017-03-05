from gensim import corpora, models, similarities
from time import time
from hazm import *
import pickle
import operator

# class MyCorpus(object):
#     def __iter__(self):
#        for line in open('/home/mahmood/PycharmProjects/Word2Vec/aaa.txt'):
#             # assume there's one document per line, tokens separated by whitespace
#             yield corpora.Dictionary.doc2bow(line.lower().split())

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# # Modeling
#
# file = '/home/mahmood/PycharmProjects/Word2Vec/sentences/cleaned.txt'
#
# class MyCorpus(object):
# 	def __iter__(self):
# 		for line in open(file):
# 			arr = []
# 			normalized_line = normalizer.normalize(line)
# 			for word in word_tokenize(normalized_line):
# 				arr.append(lemmatizer.lemmatize(word))
# 			yield dictionary.doc2bow(arr)
#
#
# def stemmedWordTokenize(line):
# 	arr = []
# 	newLine = normalizer.normalize(line)
# 	for word in word_tokenize(newLine):
# 		arr.append(lemmatizer.lemmatize(word))
# 	return arr
#
# normalizer = Normalizer()
# lemmatizer = Lemmatizer()
#
# dictionary = corpora.Dictionary(stemmedWordTokenize(line) for line in open(file))
# # print(dictionary)
# dictionary.save('obj/cleaned_stemmed/dictionary.dict')
#
# print('-------------------')
#
# corpus = MyCorpus()
# # print(corpus_memory_friendly)
#
# corpora.MmCorpus.serialize('obj/cleaned_stemmed/corpus.mm', corpus)
#
# # Transform Text with TF-IDF
# tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
# tfidf.save('obj/cleaned_stemmed/tfidf.model')
#
# # corpus tf-idf
# corpus_tfidf = tfidf[corpus]
# corpora.MmCorpus.serialize('obj/cleaned_stemmed/tfidf_corpus.mm', corpus_tfidf)


#
# # Load Dictionary and Corpus Models
dictionary = corpora.Dictionary.load('obj/cleaned_stemmed/dictionary.dict')
# print('dict loaded')
# corpus = corpora.MmCorpus('obj/aaa.mm')
#
# tfidf = models.TfidfModel.load('obj/tfidf.model')
#
corpus_tfidf = corpora.MmCorpus('obj/cleaned_stemmed/tfidf_corpus.mm')

print('Loaded')
#
# # id = 0
# # d = {dictionary.get(id): value for doc in corpus_tfidf for id, value in doc}
#
# # print(dictionary.get(41992))
# # print(d)
#
# print('-----------------')
#
# 	# print(dictionary.token2id['طریقه'])
#
# # w = 'جسور'
# # print(dictionary.get(dictionary.token2id[w]))
# # print(corpus_tfidf[1])
#
#
# # print([value for doc in corpus_tfidf for id, value in doc if id == 0])
#
d = {}
for doc in corpus_tfidf:
    for id, value in doc:
	    word = dictionary.get(id)
	    d[word] = value

with open("obj/cleaned_stemmed/words_tfidf.pkl", "wb") as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('obj/words_tfidf.pkl', 'rb') as f:
# 	d = pickle.load(f)
#
# sorted_d = sorted(d.items(), key=operator.itemgetter(1))
#
# # sorted_d = d
# # cnt = 20
# with open('words_tfidf.txt', 'w') as f:
# 	for i in reversed(sorted_d):
# 		f.write(i[0] + ' , ' + str(i[1]) + '\n')
# 		# cnt = cnt - 1
# 		# if cnt == 0:
# 		# 	break
#
