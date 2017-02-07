from os import listdir
from os.path import isfile, join
import os

orig_news_path = '/home/mahmood/PycharmProjects/sentence2vec/evaluation/files/bistoon/ijaz/system/'
news_files = [f for f in listdir(orig_news_path) if isfile(join(orig_news_path, f))]

for file in news_files: #200 News
	id = file[:-4]

	print("==> " + file)

	with open(orig_news_path + file) as f:
		old_summary = f.read().splitlines()
		print("=> " + str(len(old_summary)))
		for s, i in enumerate(old_summary):
			print(i, s)
		i = 0
		summary = ''
		flag = True
		while(len(summary.split()) < 250 and i < len(old_summary)-1):
			summary = summary + old_summary[i] + '\n'
			i += 1

	print("=+=+=+=   " + str(len(summary.split())) + "    +=+=")
	for i in summary.split():
		print(i)
	print("/-/-/-/-/-/")

	filename = '/home/mahmood/PycharmProjects/sentence2vec/evaluation/files/bistoon/ijaz_withoutCheat_files/system/' + id + '.sum'
	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	with open(filename, 'w') as g:
		g.write(summary)