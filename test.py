# import gensim
# import logging
# import mmap
# from hazm import *
# import time
# from gensim.models import Word2Vec
#
# # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# #
# # # Load Google's pre-trained Word2Vec model.
# # model = gensim.models.Word2Vec.load('models/w2v_100_combinedNormalized_cleaned.bin')
# # # model = gensim.models.Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin.gz', binary=True)
# # res = model.most_similar(positive=['امام'], topn=50)
# # # with open('university.txt', 'w') as g:
# # for r in res:
# # 	print(r)
# 		# g.write(''.join(str(r)))
# 		# g.write('\n')
#
#
# # from hazm import *
# # no = Normalizer()
# # sent1 = word_tokenize(no.normalize('قائم‌مقام مستعفی حزب اعتماد ملی با اشاره به استعفای مهدی کروبی از دبیرکلی این حزب، پیش‌بینی کرد که «همه اعضای شورای مركزی با استعفای ایشان مخالفت شدید خواهند كرد و احدی موافق این استعفا نخواهد بود».'))
# # sent2 = word_tokenize(no.normalize('امروز با كمال تأسف شنیدیم كه دبیركل محترم حزب اعتماد ملی طی نامه‌ای ضمن ابراز تذكراتی به حزب، استعفای خود را به شورای مركزی حزب اعلام كرده و درخواست كردند كه آن شورا استعفای ایشان را بپذیرد و جایگزینی را تعیین كند.'))
# # sent3 = word_tokenize(no.normalize(' با وجود اینکه روز 19 بهمن از سوی کمیته اجرایی AFC برای برگزاری بازی پلی آف لیگ قهرمانان آسیا برنامه ریزی شده بود، کمیته مسابقات لیگ برتر روز 20 بهمن را برای برگزاری دربی تعیین کرده بود.'))
# # sent4 = word_tokenize(no.normalize('در نهایت به این جمع بندی رسیدیم که دیدار دو تیم پرسپولیس و تراکتورسازی در همان تاریخ اعلام شده یعنی 17 بهمن ماه برگزار شود و در تاریخ 24 بهمن ماه دیدار دربی را برگزار خواهیم کرد.'))
# #
# # sent5 = 'Obama speaks to the media in Illinois'.lower().split()
# # sent6 = 'The president greets the press in Chicago'.lower().split()
# #
# # sent7 = word_tokenize(no.normalize('سلام عزیزم'))
# # sent8 = word_tokenize(no.normalize('سلام رفیق'))
# #
# # print(model.wmdistance(sent1, sent2))
# # print(model.wmdistance(sent3, sent4))
# # print(model.wmdistance(sent1, sent3))
# # print(model.wmdistance(sent2, sent3))
# # print(model.wmdistance(sent1, sent4))
# # print(model.wmdistance(sent2, sent4))
# #
# # print(model.wmdistance(sent5, sent6))
# #
# # print(model.wmdistance(sent7, sent8))
#
# 	# Search in Dataset
# # search_item = 'امام\u200fخمینی'
# # with open('test.txt', 'w') as g:
# # 	with open('/home/mahmood/PycharmProjects/sentence2vec/datasets/tabnak_sentences.txt') as f:
# # 		index = 0
# # 		for line in f:
# # 			index = index + 1
# # 			if search_item in line:
# # 				print(index)
# # 				print(line)
# # 				time.sleep(1)
# 				# g.write(str(index) + '\n' + line + '\n')
#
# # str = 'آنگاه فرمان انقلابی امام‏خمینی (ره) صادر شد.'
# # normalizer = Normalizer()
# # print(normalizer.normalize(str))
#
# import numpy
# import math
# from gensim import models
# #
# # dictionary = {u'media': 0, u'technolog': 3, u'accept': 4, u'inform': 2, u'model': 5, u'select': 1}
# # corpus = [[(0, 1), (1, 1)], [(2, 1), (3, 1), (4, 1)], [(0, 1), (1, 1), (5, 1)], [(2, 1), (3, 1), (4, 1), (5, 1)], [(3, 1), (4, 1)], [(0, 2)]]
# #
# # tfidf = models.TfidfModel(corpus)
# # corpus_tfidf=tfidf[corpus]
# #
# # print(dictionary)
# # print(corpus)
# #
# # d = {}
# # for doc in corpus_tfidf:
# # 	print(doc)
# # 	for id, value in doc:
# # 		print(id)
# # 		print(value)
# # 		# word = dictionary[id]
# # 	    # d[word] = value
# #
# # print(d)
#
#
#
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# corpus = [["This is very strange"],
#           ["This is very nice"]]
#
# vectorizer = TfidfVectorizer(min_df=1)
# f = list()
#
# X = vectorizer.fit_transform(corpus)
# idf = vectorizer.idf_
# dic = dict(zip(vectorizer.get_feature_names(), idf))
# f.append(dic)
# print(f)

import os

file = '/home/mahmood/Desktop/Master/Thesis/Dataset/STS/2016/sts2016-english-with-gs-v1.0/STS2016.input.question-question.txt'
result_file = '/home/mahmood/Desktop/Master/Thesis/Dataset/STS/2016/sts2016-english-with-gs-v1.0/STS2016.gs.question-question.txt'
hasResults = []
results = []

with open(result_file) as k:
    for index, line in enumerate(k):
        if line != '\n':
            hasResults.append(index)
            results.append(line)

results.reverse()

with open(file) as f:
    with open('results/STS/a_' + os.path.basename(file), 'w') as g:
        for index, line in enumerate(f):
            if index in hasResults:
                item = line.split('\t')
                g.write(item[0])
                g.write('\n')
                g.write(item[1])
                g.write('\n')
                g.write(results.pop())
                g.write('------------------------------------------')
                g.write('\n')