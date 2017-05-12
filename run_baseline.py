import pandas as pd
from sklearn.model_selection import train_test_split
import mymodel
import sys
from sklearn.metrics import log_loss, accuracy_score

RANDOM_STATE = 0
TEST_SIZE = 0.1
VAL_SIZE = 0.1


def main(train_data_with10features_path):
    df_train = pd.read_csv(train_data_with10features_path)

    # sub-sampling 1's in training set to match 1's perentage on Kaggle LB (17.5%)
    pos_boostrap_sample = \
        df_train[df_train['is_duplicate'] == 1].sample(n=55000, replace=True)
    df_train_zeros = df_train[df_train['is_duplicate'] == 0]
    rebalanced_df = pd.concat([pos_boostrap_sample, df_train_zeros])
    print 'Training data shape after rebalancing:{}'.format(rebalanced_df.shape)

    train_rebalanced_10features_path = '../dataset/train_rebalanced_10features.csv'
    rebalanced_df.to_csv(train_rebalanced_10features_path)
    print 'Saved reblanced train data to ' + train_rebalanced_10features_path +'\n'

    X = rebalanced_df.drop(['question1', 'question2', 'is_duplicate'], axis=1).values
    y = rebalanced_df['is_duplicate'].values

    # split training data to training and validation set
    print 'Splitting data into {}% training set and {}% testing set' \
        .format((1 - TEST_SIZE) * 100, TEST_SIZE * 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print 'X_train shape = {}'.format(X_train.shape)
    print 'y_train shape = {}'.format(y_train.shape)
    print 'X_test shape = {}'.format(X_test.shape)
    print 'y_test shape = {}\n'.format(y_test.shape)

    # split X_train, y_train data to training and validation set
    print 'Splitting training data into {}% real training set and {}% validation set'\
        .format((1-VAL_SIZE)*100, VAL_SIZE*100)
    X_training, X_val, y_training, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    print 'X_training shape = {}'.format(X_training.shape)
    print 'y_training shape = {}'.format(y_training.shape)
    print 'X_val shape = {}'.format(X_val.shape)
    print 'y_val shape = {}\n'.format(y_val.shape)

    # validating
    print 'Start validating...'
    RF_preds_val = mymodel.RF(X_training, y_training, X_val)

    # evaluate validation set using log loss and accuracy metrics
    val_log_loss_score = log_loss(y_val, RF_preds_val)
    print 'Validation log loss = {}'.format(val_log_loss_score)
    RF_preds_val[RF_preds_val > 0.5] = 1
    RF_preds_val[RF_preds_val <= 0.5] = 0
    accuracy = accuracy_score(y_val, RF_preds_val)
    print 'Validation accuracy = {}\n'.format(accuracy)

    # testing
    print 'Predicting...'
    RF_preds_test = mymodel.RF(X_training, y_training, X_test)

    # evaluate testing set using log loss and accuracy metrics
    test_log_loss_score = log_loss(y_test, RF_preds_test)
    print 'Testing log loss = {}'.format(test_log_loss_score)
    RF_preds_test[RF_preds_test > 0.5] = 1
    RF_preds_test[RF_preds_test <= 0.5] = 0
    accuracy = accuracy_score(y_val, RF_preds_test)
    print 'Testing accuracy = {}\n'.format(accuracy)



if __name__ == '__main__':
    train_data_with10features_path = sys.argv[1]
    main(train_data_with10features_path)
