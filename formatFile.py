from shutil import copyfile
import os

# with open('summ_news_id.csv') as f:
# 	summ_id_list = f.read().splitlines()

# PERL
# for id in summ_id_list:
# 	with open('files/ijaz_summ/'+id+'.sum') as f:
# 		lines = f.readlines()
# 	with open('files/rough/eval/models/'+id+'.html', 'w') as f:
# 		f.write('<html>\n<head><title>'+id+'</title></head>\n<body bgcolor="white">')
# 		for line in lines:
# 			f.write('<a name="'+id+'">['+id+']</a> <a href="#'+id+'" id='+id+'>'+line+'</a>\n')
# 		f.write('</body>\n</html>')

# PERL
# with open('files/rouge/eval/ROUGE-test.xml', 'w') as f:
# 	f.write('<ROUGE-EVAL version="1.0">\n<EVAL ID="1">\n<PEER-ROOT>eval/systems</PEER-ROOT>\n<MODEL-ROOT>eval/models</MODEL-ROOT>\n<INPUT-FORMAT TYPE="SEE"></INPUT-FORMAT>\n<PEERS>\n')
# 	for id in summ_id_list:
# 		f.write('<P ID="'+ id +'">'+ id +'.html</P>\n')
# 	f.write('</PEERS><MODELS>\n')
# 	for id in summ_id_list:
# 		f.write('<M ID="'+ id +'">'+ id +'.html</M>\n')
# 	f.write('</MODELS></EVAL>')

# JAVA Tabnak
# task_name = 'ijazEval1'
# for id in summ_id_list:
# 	copyfile('files/gold_summ/' + id + '.sum', 'files/java-rouge/files/system/' + task_name + '_system' + id + '.txt')
# 	# copyfile('files/gold_summ/' + id + '.sum', 'files/java-rouge/files/reference/' + task_name + '_reference' + id + '.txt')

# JAVA PASOKH - Just Rename

# path = '/home/mahmood/PycharmProjects/Word2Vec/results/w2v_pasokh_mean_withoutStop/eval/files/system/'
# for filename in os.listdir(path):
# 	os.renames(path+filename, path+'w2v-pasokh-mean-withoutStop_'+filename)

# path = '/home/mahmood/PycharmProjects/sentence2vec/evaluation/files/pasokh/optimal_files/system/'
# for filename in os.listdir(path):
# 	x = filename[19:]
# 	y = filename[:19]
# 	# print(y)
# 	os.renames(path+y+x, path+y+'.txt')

import pickle

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Save Corpus Name Dictionary
# path = '/home/mahmood/PycharmProjects/Word2Vec/results/w2v_bistoon_mean_withoutStop/eval/files/system/'
# i = 1
# names_dict = dict()
# for filename in os.listdir(path):
# 	names_dict[filename[:-4]] = i
# 	i = i + 1
# save_obj(names_dict, 'bistoon_names_dictionary')


name_ids = load_obj('pasokh_names_dictionary')
dir = '/home/mahmood/PycharmProjects/Word2Vec/results/test__w2v_pasokh_tfidf_mean_withoutStop/eval/clustering_files/'

for sys_filename in os.listdir(dir+'system/'):
	os.renames(dir+'system/'+sys_filename, dir+'system/task'+str(name_ids[sys_filename[:-4]])+'_'+sys_filename)

# for ref_filename in os.listdir(dir+'reference/'):
# 	os.renames(dir+'reference/'+ref_filename, dir+'reference/task'+str(name_ids[ref_filename[:10]+'title'])+'_'+ref_filename)


# # Add Dot end of all lines
# directory = '/home/mahmood/Desktop/Master/Thesis/Dataset/Biston/'
# for filename in os.listdir(directory + 'news/'):
# 	with open(directory + 'news/' + filename) as f:
# 		lines = f.read().splitlines()
# 		doted_lines = [line + '.' for line in lines]
# 		with open(directory + 'doted/' + filename, 'w') as g:
# 			for line in doted_lines:
# 				g.write(line)
# 				g.write('\n')
