## Paraphrase Question Identification using Feature Fusion Network 
Identify question pairs that have the same intent using Quora dataset

## Model architecture
![](https://github.com/hengluchang/SemQuestionMatching/blob/master/FFN_architecture.jpg)

## Results 
0.89 testing accuracy 

## Requirements
- Python 3.5.2

## Package dependencies
- numpy 1.11.3
- matplotlib 1.5.3
- Keras 1.2.1
- scikit-learn 0.18.1
- h5py 2.6.0
- hdf5 1.8.17

## How to run
```
$ git clone https://github.com/hengluchang/SemQuestionMatching
```

- create a folder named "dataset".
```
$ cd SemQuestionMatching
$ mkdir -p dataset
```

- Go to [Kaggle Quora Question Pairs website](https://www.kaggle.com/c/quora-question-pairs/data) and download train.csv.zip and test.csv.zip and unzip both. Place the train.csv and test.csv under /dataset directory.

- Create 10 Hand crafted features (HCFs). This will create train_10features.csv and test_10features.csv under /dataset directory.
```
$ cd ..
$ python feature_gen.py ../dataset/train.csv ../dataset/test.csv
```

# Feature Fusion Network

- Download the required data to the file you clone
- Train 

```
pyhon3 train_noHCF.py -i <QUESTION_PAIRS_FILE> -t <TEST_QUESTION_PAIRS_FILE> -g <GLOVE_FILE> -w <MODEL_WEIGHTS_FILE> -e <WORD_EMBEDDING_MATRIX_FILE> -n <NB_WORDS_DATA_FILE>
```
```
python3 train_HCF.py -i <QUESTION_PAIRS_FILE> -t <TEST_QUESTION_PAIRS_FILE> -f <HCF_FILE> -g <GLOVE_FILE> -w <MODEL_WEIGHTS_FILE> -e <WORD_EMBEDDING_MATRIX_FILE> -n <NB_WORDS_DATA_FILE>
```
- Test



- Run Random Forest basline on these 10 HCFs, this will give you ~ 0.84 testing accuracy. 

```
$ python run_baseline.py ../dataset/train_10features.csv
```


