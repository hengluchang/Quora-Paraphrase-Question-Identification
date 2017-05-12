from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def RF(X, y, X_test):
    model = RandomForestClassifier(n_estimators=500, criterion='entropy',
                                   max_depth=10, min_samples_leaf=1,
                                   max_features=0.4, n_jobs=3)
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]

    return preds


def LR(X, y, X_test):
    model = LogisticRegression()
    model.fit(X, y)
    preds = model.predict_proba(X_test)

    return preds

def XGboost(X_train, y_train, X_val, y_val, X_test):
    # Set our parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 4

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_val, label=y_val)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50,
                    verbose_eval=20)

    d_test = xgb.DMatrix(X_test)
    p_test = bst.predict(d_test)

    return p_test

