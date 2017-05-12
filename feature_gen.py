import pandas as pd
import utils
import sys

RANDOM_STATE = 0


def main(train_data_path, test_data_path):

    df_train = pd.read_csv(train_data_path)
    # drop unused columns in training data
    df_train = df_train.drop(['id', 'qid1', 'qid2'], axis=1)

    # add features to training data
    print 'Start engineering 10 HCFs for training data...'
    print 'This might take a while...'
    df_train = utils.feature_eng(df_train)


    print df_train.head()

    # update train.csv with new features as columns
    train_10features_path = '../dataset/train_10features.csv'
    df_train.to_csv(train_10features_path, index=False)
    print 'Finish engineering 10 HCFs for training data and save in ' + \
          train_10features_path + '\n'

    # add features to testing data
    print 'Loading test data....'
    df_test = pd.read_csv(test_data_path)
    print 'Start engineering 10 HCFs for testing data...'
    print 'This will take even longer...'
    df_test = utils.feature_eng(df_test)

    # save new features to testing data
    test_10features_path = '../dataset/test_10features.csv'
    df_test.to_csv(test_10features_path, index=False)
    print 'Finish engineering 10 HCFs for Kaggle testing data and save in ' + \
          test_10features_path + '\n'

if __name__ == '__main__':
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    main(train_data_path, test_data_path)





