## Adapted from https://github.com/bradleypallen/keras-quora-question-pairs
## Po-Cheng Pan

from __future__ import print_function
import numpy as np
import csv, datetime, time, json, getopt, sys
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda, Concatenate, Merge
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

KERAS_DATASETS_DIR = expanduser('~/.keras/datasets/')
QUESTION_PAIRS_FILE_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
GLOVE_ZIP_FILE = 'glove.840B.300d.zip'
Q1_TRAINING_DATA_FILE = 'q1_train_rebalanced.npy'
Q2_TRAINING_DATA_FILE = 'q2_train_rebalanced.npy'
TRAINING_HCF_FILE_URL = 'https://drive.google.com/open?id=0Bx-K09TZawidN3NubW9ia093Nlk'
TRAINING_HCF_FILE = 'HCF_train_rebalanced.npy'
Q1_TESTING_DATA_FILE = 'q1_test.npy'
Q2_TESTING_DATA_FILE = 'q2_test.npy'
LABEL_TRAINING_DATA_FILE = 'label_train.npy'
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 13371448
NB_EPOCHS = 100

def main(argv):

    ## Define custon metrics
    def f1_score(y_true, y_pred):

        # Count positive samples.
        c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
        c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

        # If there are no true samples, fix the F1 score at 0.

        # How many selected items are relevant?
        precision = c1 / (c2 + K.epsilon())

        # How many relevant items are selected?
        recall = c1 / (c3 + K.epsilon())

        # Calculate f1_score
        f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_score

    def precision(y_true, y_pred):

        # Count positive samples.
        c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))

        # How many selected items are relevant?
        precision = c1 / (c2+K.epsilon())

        return precision

    def recall(y_true, y_pred):

        # Count positive samples.
        c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

        # How many relevant items are selected?
        recall = c1 / (c3+K.epsilon())

        return recall

    ## Parse input arguments
    try:
        opts, args = getopt.getopt(argv,"hi:t:f:g:w:e:n:",["ifile=", "tfile= ", "hcffile", "gfile=","wfile=", "embeddingfile", "nbfile"])
    except getopt.GetoptError: 
        print ('train_HCF.py -i <QUESTION_PAIRS_FILE> -t <TEST_QUESTION_PAIRS_FILE> -f <HCF_FILE> -g <GLOVE_FILE> -w <MODEL_WEIGHTS_FILE> -e <WORD_EMBEDDING_MATRIX_FILE> -n <NB_WORDS_DATA_FILE>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print ('train_HCF.py -i <QUESTION_PAIRS_FILE> -t <TEST_QUESTION_PAIRS_FILE> -f <HCF_FILE> -g <GLOVE_FILE> -w <MODEL_WEIGHTS_FILE> -e <WORD_EMBEDDING_MATRIX_FILE> -n <NB_WORDS_DATA_FILE>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            QUESTION_PAIRS_FILE = arg
        elif opt in ("-t", "--tfile"):
            TEST_QUESTION_PAIRS_FILE = arg
        elif opt in ("-f", "--hcffile"):
            HCF_FILE = arg
        elif opt in ("-g", "--gfile"):
            GLOVE_FILE = arg
        elif opt in ("-w", "--wfile"):
            MODEL_WEIGHTS_FILE = arg
        elif opt in ("-e", "--embeddingfile"):
            WORD_EMBEDDING_MATRIX_FILE = arg
        elif opt in ("-n", "--nbfile"):
            NB_WORDS_DATA_FILE = arg

    if exists(Q1_TRAINING_DATA_FILE) and exists(Q2_TRAINING_DATA_FILE) and exists(LABEL_TRAINING_DATA_FILE) and exists(NB_WORDS_DATA_FILE) and exists(WORD_EMBEDDING_MATRIX_FILE) and exists(TRAINING_HCF_FILE):
        q1_data = np.load(open(Q1_TRAINING_DATA_FILE, 'rb'))
        q2_data = np.load(open(Q2_TRAINING_DATA_FILE, 'rb'))
        labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))
        HCF_data = np.load(open(TRAINING_HCF_FILE, 'rb'))
        word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
        with open(NB_WORDS_DATA_FILE, 'r') as f:
            nb_words = json.load(f)['nb_words']
    else:

        print("Processing", TRAINING_HCF_FILE)
        q1_word_count = []
        q2_word_count = []
        word_count_diff = []
        word_overlap = []
        uni_BLEU = []
        bi_BLEU = []
        BLEU2 = []
        char_bigram_overlap = []
        char_trigram_overlap = []
        char_4gram_overlap = []

        with open(HCF_FILE, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                q1_word_count.append(row['q1_word_count'])
                q2_word_count.append(row['q2_word_count'])
                word_count_diff.append(row['word_count_diff'])
                word_overlap.append(row['word_overlap'])
                uni_BLEU.append(row['uni_BLEU'])
                bi_BLEU.append(row['bi_BLEU'])
                BLEU2.append(row['BLEU2'])
                char_bigram_overlap.append(row['char_bigram_overlap'])
                char_trigram_overlap.append(row['char_trigram_overlap'])
                char_4gram_overlap.append(row['char_4gram_overlap'])

        HCF_data = np.array([q1_word_count, q2_word_count, word_count_diff, word_overlap, uni_BLEU, bi_BLEU, BLEU2, char_bigram_overlap, char_trigram_overlap, char_4gram_overlap])


        print("Processing", QUESTION_PAIRS_FILE)

        question1 = []
        question2 = []
        question1_test = []
        question2_test = []
        is_duplicate = []
        with open(QUESTION_PAIRS_FILE, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                question1.append(row['question1'])
                question2.append(row['question2'])
                is_duplicate.append(row['is_duplicate'])

        print('Question pairs: %d' % len(question1))

        with open(TEST_QUESTION_PAIRS_FILE, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                question1_test.append(row['question1'])
                question2_test.append(row['question2'])



        questions = question1 + question2 + question1_test + question2_test
        tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(questions)
        question1_word_sequences = tokenizer.texts_to_sequences(question1)
        question2_word_sequences = tokenizer.texts_to_sequences(question2)
        word_index = tokenizer.word_index

        print("Words in index: %d" % len(word_index))
        
        if not exists(GLOVE_ZIP_FILE) or not exists(GLOVE_FILE):
            zipfile = ZipFile(get_file(GLOVE_ZIP_FILE, GLOVE_ZIP_FILE_URL))
            zipfile.extract(GLOVE_FILE, path=KERAS_DATASETS_DIR)

        print("Processing", GLOVE_FILE)

        embeddings_index = {}
        with open(GLOVE_FILE, encoding='utf-8') as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = embedding

        print('Word embeddings: %d' % len(embeddings_index))

        nb_words = min(MAX_NB_WORDS, len(word_index))
        word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i > MAX_NB_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                word_embedding_matrix[i] = embedding_vector
            
        print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

        q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        labels = np.array(is_duplicate, dtype=int)
        print('Shape of question1 data tensor:', q1_data.shape)
        print('Shape of question2 data tensor:', q2_data.shape)
        print('Shape of label tensor:', labels.shape)

        np.save(open(Q1_TRAINING_DATA_FILE, 'wb'), q1_data)
        np.save(open(Q2_TRAINING_DATA_FILE, 'wb'), q2_data)
        np.save(open(LABEL_TRAINING_DATA_FILE, 'wb'), labels)
        np.save(open(WORD_EMBEDDING_MATRIX_FILE, 'wb'), word_embedding_matrix)
        np.save(open(TRAINING_HCF_FILE, 'wb'), HCF_data)
        with open(NB_WORDS_DATA_FILE, 'w') as f:
            json.dump({'nb_words': nb_words}, f)

    print(q1_data.shape)
    print(HCF_data.shape)
    
    X = np.stack((q1_data, q2_data, HCF_data), axis=1)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)
    Q1_train = X_train[:,0]
    Q2_train = X_train[:,1]
    HCF_train = np.transpose(np.transpose(X_train[:,2])[1:11])
    print(HCF_train.shape)
    Q1_test = X_test[:,0]
    Q2_test = X_test[:,1]
    HCF_test = np.transpose(np.transpose(X_test[:,2])[1:11])

    Q1 = Sequential()
    Q1.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    Q1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
    Q1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM, )))
    Q2 = Sequential()
    Q2.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    Q2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
    Q2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM, )))
    HCF = Sequential()
    #HCF.add(Dense(10, input_shape = (10,)))
    HCF.add(Reshape((10,), input_shape=(10,)))

    model = Sequential()


    model.add(Merge([Q1, Q2, HCF], mode='concat'))
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
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

    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]

    print("Starting training at", datetime.datetime.now())

    t0 = time.time()
    history = model.fit([Q1_train, Q2_train, HCF_train], 
                        y_train, 
                        epochs=NB_EPOCHS, 
                        validation_split=VALIDATION_SPLIT,
                        #validation_data = ([Q1_test, Q2_test], y_test), 
                        verbose=1, 
                        callbacks=callbacks)
    t1 = time.time()

    print("Training ended at", datetime.datetime.now())

    print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

    # list all data in history
    print("All data in history: ",history.history.keys())
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model(with HCF) accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig.savefig('model_HCF_accuracy.png')
    #plt.show()

    # summarize history for loss
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model(with HCF) loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig.savefig('model_HCF_loss.png')
    #plt.show()

    # summarize history for precision
    fig = plt.figure()
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('model(with HCF) precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig.savefig('model_HCF_precision.png')


    # summarize history for recall
    fig = plt.figure()
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('model(with HCF) recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig.savefig('model_HCF_recall.png')

    # summarize history for f1 score
    fig = plt.figure()
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.title('model(with HCF) f1_score')
    plt.ylabel('f1_score')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig.savefig('model_HCF_f1_score.png')

    model.load_weights(MODEL_WEIGHTS_FILE)

    print('Testing Data Metrics:')
    loss, accuracy, precision, recall, f1_score = model.evaluate([Q1_test, Q2_test, HCF_test], y_test)
    print('')
    print('loss      = {0:.4f}'.format(loss))
    print('accuracy  = {0:.4f}'.format(accuracy))
    print('precision = {0:.4f}'.format(precision))
    print('recall    = {0:.4f}'.format(recall))
    print('F         = {0:.4f}'.format(f1_score))
    plot_model(model, to_file='model.png')

if __name__ == "__main__":
  main(sys.argv[1:])