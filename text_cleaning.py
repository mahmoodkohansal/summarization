import fileinput

input_file = 'sentences/combined_sentences.txt'
output_file = 'datasets/combined_sentences_normalized.txt'
ignore_list = ['&nbsp؛', '\u200c', '\u200f', '،', '.', '(', ')', '[', ']', '«', '»', '*', '؛', ':', '/', '\\', '؟', '!']

with open(input_file) as f:
	with open(output_file, 'w') as g:
		index = 0
		for line in f:
			index = index + 1
			for ign in ignore_list:
				line = line.replace(ign, '')
			g.write(line)

# with open(output_file) as f:
# 	counter = 0
# 	for line in f:
# 		if '&nbsp' in line:
# 			counter = counter + 1
# print(counter)