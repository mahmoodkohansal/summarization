import nltk
import string
import os
from hazm import *
from textblob import TextBlob as tb

from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.stem.porter import PorterStemmer

path = '/home/mahmood/PycharmProjects/Word2Vec/sentences'
token_dict = {}
# stemmer = PorterStemmer()
normalizer = Normalizer()


# def stem_tokens(tokens, stemmer):
# 	stemmed = []
# 	for item in tokens:
# 		stemmed.append(stemmer.stem(item))
# 	return stemmed


def tokenize(text):
	text_normalized = normalizer.normalize(text)
	tokens = word_tokenize(text_normalized)
	# stems = stem_tokens(tokens, stemmer)
	return tokens


for subdir, dirs, files in os.walk(path):
	for file in files:
		file_path = subdir + os.path.sep + file
		shakes = open(file_path, 'r')
		text = shakes.read()
		lowers = text
		remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
		no_punctuation = lowers.translate(remove_punctuation_map)
		token_dict[file] = no_punctuation

print(token_dict)

# this can take some time
tfidf = TfidfVectorizer()
# print(tfidf)

tfs = tfidf.fit_transform(token_dict.values())
print(tfs)
dense = tfs.todense()
episode = dense[0].tolist()[0]
phrase_scores = [pair for pair in zip(range(0, len(episode)), episode)]
zz = sorted(phrase_scores, key=lambda t: t[1] * -1)
print('\n--- TFIDF ----\n')
print(zz)
print('\n-->>\n')
feature_names = tfidf.get_feature_names()
for z in zz:
	print(feature_names[z[0]], z[1])

# print('\n----IDF---\n')
#
# xx = tfidf._tfidf.idf_
# print(xx)
#
# print('\n----Features Count---\n')
# yy = tfidf.get_feature_names()
# print(len(yy))
# print('\n----Feature Name + TFIDF? ---\n')
# print(dict(zip(tfidf.get_feature_names(), xx)))
# print('\n-------\n')
# for i in tfs:
# 	print(i)

print('===================')

# str = 'سلام من به تو یار قدیمی، منم همان هوادار قدیمی، است روز اوضاع'
# response = tfidf.transform([str])
# print(response)
#
# feature_names = tfidf.get_feature_names()
# for col in response.nonzero()[1]:
#     print(feature_names[col], ' - ', response[0, col])

# import math
# from textblob import TextBlob as tb
# import os
# from hazm import *
#
# def tf(word, blob):
#     return blob.words.count(word) / len(blob.words)
#
# def n_containing(word, bloblist):
#     return sum(1 for blob in bloblist if word in blob.words)
#
# def idf(word, bloblist):
#     return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))
#
# def tfidf(word, blob, bloblist):
#     return tf(word, blob) * idf(word, bloblist)
#
# input_file = '/home/mahmood/PycharmProjects/Word2Vec/sentences/bbb.txt'
#
# blob_list = list()
#
# with open(input_file) as f:
# 	file_string = f.read()
# 	sentences = sent_tokenize(file_string)
#
# 	for sent in sentences:
# 		blob_list.append(tb(sent))
#
# for i in blob_list:
# 	print(i.words)

# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(min_df=1)
# corpus = [
#      'This is the first document.',
#      'This is the second second document.',
#      'And the third one.',
#      'Is this the first document?',
# ]
#
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.vocabulary_)

