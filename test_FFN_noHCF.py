## Written by Po-Cheng Pan
## Test.py 

from __future__ import print_function
import numpy as np
import csv, datetime, time, json, sys, getopt
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split



def main(argv):

	def f1_score(y_true, y_pred):

	    # Count positive samples.
	    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
	    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

	    # If there are no true samples, fix the F1 score at 0.
	    if c3 == 0 or c2 == 0:
	        return 0

	    # How many selected items are relevant?
	    precision = c1 / c2

	    # How many relevant items are selected?
	    recall = c1 / c3

	    # Calculate f1_score
	    f1_score = 2 * (precision * recall) / (precision + recall)
	    return f1_score

	def precision(y_true, y_pred):

	    # Count positive samples.
	    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))

	    if c2 == 0:
	    	return 0
	    # How many selected items are relevant?
	    precision = c1 / c2

	    return precision

	def recall(y_true, y_pred):

	    # Count positive samples.
	    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

	    # If there are no true samples, fix the F1 score at 0.
	    if c3 == 0:
	        return 0

	    # How many relevant items are selected?
	    recall = c1 / c3

	    return recall

	try:
	    opts, args = getopt.getopt(argv,"hi:o:e:n:w:",["ifile=", "ofile=", "embeddingfile", "nbfile", "weightFile"])
	except getopt.GetoptError: 
		print ('test_FFN_noHCF.py -i <QUESTION_PAIRS_FILE> -o <RESULT_FILE> -e <WORD_EMBEDDING_MATRIX_FILE> -n <NB_WORDS_DATA_FILE> -w <MODEL_WEIGHTS_FILE>')
		sys.exit(2)
	    
	for opt, arg in opts:
		if opt == '-h':
			print ('test_FFN_noHCF.py -i <QUESTION_PAIRS_FILE> -o <RESULT_FILE> -e <WORD_EMBEDDING_MATRIX_FILE> -n <NB_WORDS_DATA_FILE> -w <MODEL_WEIGHTS_FILE>')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			QUESTION_PAIRS_FILE = arg
		elif opt in ("-o", "--ofile"):
			RESULT_FILE = arg
		elif opt in ("-e", "--embeddingfile"):
			WORD_EMBEDDING_MATRIX_FILE = arg
		elif opt in ("-n", "--nbfile"):
			NB_WORDS_DATA_FILE = arg
		elif opt in ("-w", "--weightFile"):
			MODEL_WEIGHTS_FILE = arg

	## Initialize global variables
	EMBEDDING_DIM = 300
	MAX_SEQUENCE_LENGTH = 25
	MAX_NB_WORDS = 200000
	Q1_TESTING_DATA_FILE = 'q1_test.npy'
	Q2_TESTING_DATA_FILE = 'q2_test.npy'
	## Load word embedding matrix
	word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
	## Load nb words data
	with open(NB_WORDS_DATA_FILE, 'r') as f:
		nb_words = json.load(f)['nb_words']

	## Load testing question pairs
	if exists(Q1_TESTING_DATA_FILE) and exists(Q2_TESTING_DATA_FILE) and exists(WORD_EMBEDDING_MATRIX_FILE) and exists(NB_WORDS_DATA_FILE):
	    q1_data = np.load(open(Q1_TESTING_DATA_FILE, 'rb'))
	    q2_data = np.load(open(Q2_TESTING_DATA_FILE, 'rb'))
	else:
		print("Processing", QUESTION_PAIRS_FILE)

		question1 = []
		question2 = []
		with open(QUESTION_PAIRS_FILE, encoding='utf-8') as csvfile:
			reader = csv.DictReader(csvfile, delimiter=',')
			for row in reader:
				question1.append(row['question1'])
				question2.append(row['question2'])
	        	
		print('Question pairs: %d' % len(question1))

		questions = question1 + question2
		tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
		tokenizer.fit_on_texts(questions)
		question1_word_sequences = tokenizer.texts_to_sequences(question1)
		question2_word_sequences = tokenizer.texts_to_sequences(question2)
		word_index = tokenizer.word_index

		print("Words in index: %d" % len(word_index))
		
		q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
		q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
		print('Shape of question1 data tensor:', q1_data.shape)
		print('Shape of question2 data tensor:', q2_data.shape)

		np.save(open(Q1_TESTING_DATA_FILE, 'wb'), q1_data)
		np.save(open(Q2_TESTING_DATA_FILE, 'wb'), q2_data)

	X = np.stack((q1_data, q2_data), axis=1)
	Q1_test = X[:,0]
	Q2_test = X[:,1]

	Q1 = Sequential()
	Q1.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
	Q1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
	Q1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM, )))
	Q2 = Sequential()
	Q2.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
	Q2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
	Q2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM, )))

	## Build the model
	print("Build Model")
	model = Sequential()
	model.add(Merge([Q1, Q2], mode='concat'))
	model.add(BatchNormalization())
	model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy', precision, recall, f1_score])

	print("Load Weight Matrix")
	model.load_weights(MODEL_WEIGHTS_FILE)

	print("Predict labels")
	result = model.predict([Q1_test, Q2_test], 128, 1)

	f = open(RESULT_FILE,'w')
	f.write('test_id,is_duplicate\n')
	id = 0
	for label in np.nditer(result):
		f.write(str(id)+','+ str(label) +'\n')
		id += 1
	f.close()
	print('')
if __name__ == "__main__":
  main(sys.argv[1:])
