from hazm import *

filename = 'sentences/cleaned.txt'

stemmer = Stemmer()
lemmatizer = Lemmatizer()

for line in open(filename):
	for word in word_tokenize(line):
		if word != lemmatizer.lemmatize(word):
			print(word, stemmer.stem(word), lemmatizer.lemmatize(word))