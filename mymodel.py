from sklearn.ensemble import RandomForestClassifier


def RF(X, y, X_test):
    model = RandomForestClassifier(n_estimators=500, criterion='entropy',
                                   max_depth=10, min_samples_leaf=1,
                                   max_features=0.4, n_jobs=3)
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]

    return preds



