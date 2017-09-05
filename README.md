[![Codacy Badge](https://api.codacy.com/project/badge/Grade/086fe025b4fb41599ee1e6dfa50f12bf)](https://www.codacy.com/app/hengluchang/Quora-Paraphrase-Question-Identification?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hengluchang/Quora-Paraphrase-Question-Identification&amp;utm_campaign=Badge_Grade)

## Paraphrase Question Identification using Feature Fusion Network 
Identify question pairs that have the same meaning. Feature Fusion Network takes advantage of learning rich features not just from sentence representations but also from hand craft features.  

For more detailed information, please see our project research paper: [Paraphrase Question Identification using Feature Fusion Network](https://github.com/hengluchang/Quora-Paraphrase-Question-Identification/blob/master/paraphrase-question-identification.pdf). 

## Model architecture
![](https://github.com/hengluchang/SemQuestionMatching/blob/master/FFN_architecture.jpg)

## Results 
- 0.895 testing accuracy for FFN (train for 100 epoch)

## Requirements
- Python 3.5 for running FFN
- Python 2.7 for running Random Forest (RF) baseline 

## Package dependencies
### RF baseline
- scikit-learn 0.18
- nltk
- pandas 

### FFN
- numpy 1.11
- matplotlib 1.5
- Keras 1.2
- scikit-learn 0.18
- h5py 2.6
- hdf5 1.8
- TensorFlow 0.10

## How to run
```
$ git clone https://github.com/hengluchang/Quora-Paraphrase-Question-Identification
```
### Run Random Forest baseline 
- create a folder named "dataset".
```
$ cd Quora-Paraphrase-Question-Identification
$ mkdir -p dataset
```

- Go to [Kaggle Quora Question Pairs website](https://www.kaggle.com/c/quora-question-pairs/data) and download train.csv.zip and test.csv.zip and unzip both. Place the train.csv and test.csv under /dataset directory.

- Create 10 Hand crafted features (HCFs). This will create train_10features.csv and test_10features.csv under /dataset directory.
```
$ cd ..
$ python feature_gen.py ../dataset/train.csv ../dataset/test.csv
```

- Run Random Forest baseline on these 10 HCFs, this will give you ~ 0.84 testing accuracy. 

```
$ python run_baseline.py ../dataset/train_10features.csv
```

### Run Feature Fusion Network (FFN)

- Download the required data [here(Google Drive link)](https://drive.google.com/drive/folders/0B7j2V-uXleQ-ZjhxS0laWFBBTVk?usp=sharing) to the directory you clone

- Train FFN w/o HCF
```
$ pyhon3 train_noHCF.py -i <QUESTION_PAIRS_FILE> -t <TEST_QUESTION_PAIRS_FILE> -g <GLOVE_FILE> -w <MODEL_WEIGHTS_FILE> -e <WORD_EMBEDDING_MATRIX_FILE> -n <NB_WORDS_DATA_FILE>
```
For instance:
```
$ python3 train_noHCF.py -i train_rebalanced.csv -t test.csv -g glove.840B.300d.txt -w question_pairs_weights_100epoch_test10_val10_dropout20_sumOP_noAVG_rebalanced.h5  -e word_embedding_matrix_trainANDtest_rebalanced.npy -n nb_words_trainANDtest_rebalanced.json
```

- Train FFN
```
$ python3 train_HCF.py -i <QUESTION_PAIRS_FILE> -t <TEST_QUESTION_PAIRS_FILE> -f <HCF_FILE> -g <GLOVE_FILE> -w <MODEL_WEIGHTS_FILE> -e <WORD_EMBEDDING_MATRIX_FILE> -n <NB_WORDS_DATA_FILE>
```
For instance: 
```
$ python3 train_HCF.py -i train_rebalanced.csv -t test.csv -f train_rebalanced_10features.csv -g glove.840B.300d.txt -w question_pairs_weights_100epoch_test10_val20_dropout20_sumOP_noAVG_HCF_rebalanced.h5  -e word_embedding_matrix_trainANDtest_rebalanced.npy -n nb_words_trainANDtest_rebalanced.json
```

- Test FFN w/o HCF
```
$ python3 test_noHCF.py -i <QUESTION_PAIRS_FILE> -o <RESULT_FILE> -e <WORD_EMBEDDING_MATRIX_FILE> -n <NB_WORDS_DATA_FILE> -w <MODEL_WEIGHTS_FILE>
```
For instance:
```
$ python3 test_noHCF.py -i test.csv  -o result_question_pairs_weights_100epoch_test10_val10_dropout20_sumOP_noAVG_rebalanced.csv -e word_embedding_matrix_trainANDtest_rebalanced.npy -n nb_words_trainANDtest_rebalanced.json -w question_pairs_weights_100epoch_test10_val10_dropout20_sumOP_noAVG_rebalanced.h5 
```
- Test FFN

```
$ python3 test_HCF.py -i <QUESTION_PAIRS_FILE> -o <RESULT_FILE> -e <WORD_EMBEDDING_MATRIX_FILE> -n <NB_WORDS_DATA_FILE> -w <MODEL_WEIGHTS_FILE>
```
For instance:
```
$ python3 test_sum_HCF.py -i test.csv -f -test_10features.csv -o result_question_pairs_weights_100epoch_test10_val10_dropout20_sumOP_noAVG_HCF_rebalanced.csv -e word_embedding_matrix_trainANDtest_rebalanced.npy -n nb_words_trainANDtest_rebalanced.json -w question_pairs_weights_100epoch_test10_val10_dropout20_sumOP_noAVG_HCF_rebalanced.h5
```

## Reference
- [Keras model to identify Quora question pairs](https://github.com/bradleypallen/keras-quora-question-pairs): borrowed most of the Deep Neural Network script



