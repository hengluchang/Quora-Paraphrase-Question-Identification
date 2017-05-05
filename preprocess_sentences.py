## Preprocess the csv file
import csv, sys, getopt, re

def main(argv):
	# Initialize the parameters
	inputfile = ''
	outputfile = ''
	q1 = ''
	q2 = ''
	line = 0
	output = ''


	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print 'Preprocess_csv.py -i <inputfile> -o <outputfile>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'test.py -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg
	print 'Input file is "', inputfile
	print 'Output file is "', outputfile

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

			q1 = re.sub(' +',' ', q1)
			q2 = re.sub(' +',' ', q2)

			f.write(q1+'\n') 
			f.write(q2+'\n') 


	print len(output)
	## Python will convert \n to os.linesep
	f.close()

if __name__ == "__main__":
	main(sys.argv[1:])