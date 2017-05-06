## Written by Po-Cheng Pan
## Preprocess input data into word2vec by gensim

import gensim
import sys, getopt

def main(argv):

	inputfile = ''
	outputfile = ''

	try:
	    opts, args = getopt.getopt(argv,"hi:o:",["ifile=", "ofile="])
	except getopt.GetoptError: 
		print 'word2vec_gensim.py -i <inputfile> -o <outputfile>'
		sys.exit(2)
	    
	for opt, arg in opts:
		if opt == '-h':
			print 'word2vec_gensim.py -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg
	print 'Input file is ', inputfile 
	print 'Output file is ', outputfile

	#filename = "./Data/train_sentences"
	with open(inputfile) as f:
		content = f.read().splitlines()

	sentences = []
	for line in content:
		sentences.append(line.strip().split())

	model = gensim.models.Word2Vec(sentences, size=300, window=5, min_count=3, workers=4)

	model.save(outputfile)
	#model.save_word2vec_format(outputfile)



if __name__ == "__main__":
  main(sys.argv[1:])