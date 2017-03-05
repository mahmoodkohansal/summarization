import os
import shutil
import pickle
import subprocess
import pandas as pd
import numpy

def func1(directory):
	dataset = 'pasokh'
	rougeResourcePath = '/home/mahmood/Desktop/Master/Thesis/Resources/rouge/'
	refResourcePath = '/home/mahmood/Desktop/Master/Thesis/Resources/reference/'

	# copy required files
	if os.path.isfile(directory + 'rouge.properties'):
		pass
	else:
		for file in os.listdir(rougeResourcePath):
			shutil.copy2(rougeResourcePath + file, directory)
		os.makedirs(directory + 'results/')

	# load dataset task names
	if dataset == 'pasokh':
		with open('obj/pasokh_names_dictionary.pkl', 'rb') as f:
			name_ids = pickle.load(f)

	for item in os.listdir(directory + 'files/'):
		# system files
		for sys_filename in os.listdir(directory + 'files/' + item + '/system/'):
			os.renames(directory + 'files/' + item + '/system/' + sys_filename,
			           directory + 'files/' + item + '/system/task' + str(name_ids[sys_filename[:-4]]) + '_' + sys_filename)
		# references files
		shutil.copytree(refResourcePath, directory + 'files/' + item + '/reference/')

		for i in range(1, 5):
			# prepare rouge properties
			with open(directory + 'rouge.properties', 'w') as h:
				h.write('project.dir=files/' + item + '\n')
				h.write('rouge.type=normal\n')
				h.write('ngram=' + str(i) + '\n')
				h.write('stopwords.use=false\n')
				h.write('stopwords.file=resources/stopwords-rouge-default.txt\n')
				h.write('topic.type=nn|jj\n')
				h.write('synonyms.use=false\n')
				h.write('synonyms.dir=default\n')
				h.write('pos_tagger_name=english-bidirectional-distsim.tagger\n')
				h.write('output=file\n')
				h.write('outputFile=results/r' + str(i) + '-' + item + '.csv\n')

			# calculate rouge
			# subprocess.run(["java", "-jar ", directory + "rouge2.0.jar"])
			subprocess.call('cd ' + directory + '; java -jar rouge2.0.jar', shell=True)


def func2(file):
	df = pd.read_csv(file)

	avg_recall = numpy.mean(df.Avg_Recall)
	avg_precision = numpy.mean(df.Avg_Precision)
	avg_f = numpy.mean(df['Avg_F-Score'])

	print(avg_recall, avg_precision, avg_f)
	with open(file, 'a') as f:
		f.write(',,,' + str(avg_recall) + ',' + str(avg_precision) + ',' + str(avg_f) + '\n')

# func1('/home/mahmood/PycharmProjects/Word2Vec/results/w2v_300_stemmed/w2v_pasokh_tfidf_mean_STEM_withStop/eval/')
func2('/home/mahmood/PycharmProjects/Word2Vec/results/w2v_300_stemmed/w2v_pasokh_tfidf_mean_STEM_withStop/eval/results/r1-textrank.csv')