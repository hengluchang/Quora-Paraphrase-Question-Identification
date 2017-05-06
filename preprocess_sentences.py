## Written by Po-Cheng Pan
## Preprocess the csv file
import csv, sys, getopt, re, collections, string, time, sets

def main(argv):
	# Initialize the parameters
	inputfile = ''
	outputfile = ''
	q1 = ''
	q2 = ''
	line = 0
	output = ''


	try:
		opts, args = getopt.getopt(argv,"hi:o:v:",["ifile=","ofile=", "vsize"])
	except getopt.GetoptError:
		print 'preprocess_sentences.py -i <inputfile> -o <outputfile> -v <vocabulary_size>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'preprocess_sentences.py -i <inputfile> -o <outputfile> -v <vocabulary_size>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg
		elif opt in ("-v", "--vsize"):
			vocabulary_size = string.atoi(arg)
	print 'Input file is "', inputfile
	print 'Output file is "', outputfile
	print 'Vocabulary size is ', vocabulary_size

	#f = open(outputfile,'w')
	q = []

	with open(inputfile, 'rb') as csvfile:
		spamreader = csv.reader(csvfile)
		for row in spamreader:
			

			# Replace numbers with a word "num"
			q1 = re.sub('\&', 'and', row[3])
			q2 = re.sub('\&', 'and', row[4])

			# Convert strings to lowercase
			q1 = q1.lower()
			q2 = q2.lower()

			q1 = re.sub('[^0-9a-zA-Z ]+', ' ', q1)
			q2 = re.sub('[^0-9a-zA-Z ]+', ' ', q2)

			q1 = re.sub(' +',' ', q1).strip()
			q2 = re.sub(' +',' ', q2).strip()

			#qInOneLine = ' '.join([qInOneLine,q1, q2])
			q.extend(q1.strip().split())
			q.extend(q2.strip().split())
			

			
			#f.write(q1+'\n') 
			#f.write(q2+'\n') 
	c = sets.Set(collections.Counter(q).most_common(vocabulary_size - 1))
	set_MostCommon = sets.Set()
	for word, _ in c:
		set_MostCommon.add(word)

	
	f = open(outputfile,'w')
	with open(inputfile, 'rb') as csvfile:
			spamreader = csv.reader(csvfile)
			for row in spamreader:
				

				# Replace numbers with a word "num"
				q1 = re.sub('\&', 'and', row[3])
				q2 = re.sub('\&', 'and', row[4])

				# Convert strings to lowercase
				q1 = q1.lower()
				q2 = q2.lower()

				q1 = re.sub('[^0-9a-zA-Z ]+', ' ', q1)
				q2 = re.sub('[^0-9a-zA-Z ]+', ' ', q2)

				q1 = re.sub(' +',' ', q1).strip()
				q2 = re.sub(' +',' ', q2).strip()

				q1_new = ''
				for w in q1.split():
					if w in set_MostCommon:
						q1_new+=(w+' ')
					else:
						q1_new+="UNK "
				q2_new = ''
				for w in q2.split():
					if w in set_MostCommon:
						q2_new+=(w+' ')
					else:
						q2_new+="UNK "
				f.write(q1_new+'\n') 
				f.write(q2_new+'\n')

				
	#print len(output)
	## Python will convert \n to os.linesep
	f.close()

if __name__ == "__main__":
	main(sys.argv[1:])