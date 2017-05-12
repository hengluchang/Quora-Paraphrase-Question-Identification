## Paraphrase Question Identification using Feature Fusion Network 
Identify question pairs that have the same intent using Quora dataset

## Results 
0.89 testing accuracy 

## How to run
```
$ git clone https://github.com/hengluchang/SemQuestionMatching
```
-create a folder named "dataset".
```
$ cd SemQuestionMatching
$ mkdir -p dataset
```

-Go to [Kaggle Quora Question Pairs website](https://www.kaggle.com/c/quora-question-pairs/data) and download train.csv.zip and test.csv.zip and unzip both. Place the train.csv and test.csv under /dataset directory.

- Create 10 Hand crafted feature. This will create train_10features.csv and test_10features.csv under /dataset directory.
```
$ cd ..
$ python feature_gen.py ../dataset/train.csv ../dataset/test.csv
```

- Run Random Forest basline on these 10 HCFs, this will give you ~ 0.84 testing accuracy. 

```
$ python run_baseline.py ../dataset/train_10features.csv
```

- Run Random Forest baseline  

