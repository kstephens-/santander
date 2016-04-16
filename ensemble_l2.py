import pandas as pd
import numpy as np
import xgboost as xgb

import features

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer, Binarizer, scale
from sklearn.feature_selection import SelectPercentile, f_classif, chi2

from unbalanced_dataset import UnderSampler, TomekLinks, SMOTE

from scipy.stats.mstats import gmean, hmean

submit = False
version = '2.02'

def xgboost_model(train, labels, test):

    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'auc'

    params['eta'] = 0.001
    params['gamma'] = 1.0
    params['max_depth'] = 3
    params['min_child_weight'] = 15
    params['max_delta_step'] = 0
    params['subsample'] = 0.2
    params['colsample_bytree'] = 1.0
    params['lambda'] = 1
    params['alpha'] = 0
    params['seed'] = 4

    params['silent'] = 1

    xgtrain = xgb.DMatrix(train, labels, missing=9999999999)
    xgtest = xgb.DMatrix(test, missing=9999999999)

    num_rounds = 3600
    m = xgb.train(list(params.items()), xgtrain, num_rounds)
    return m, m.predict(xgtest)


data_train = pd.read_csv('../ensemble/train_{}.csv'.format(version))
y = data_train['TARGET']

if submit:
    data_test = pd.read_csv('../ensemble/test_{}.csv'.format(version))
    test_idx = data_test.ID.values
    submission = pd.DataFrame()

    data_test.drop(['ID'], axis=1, inplace=True)
    data_test.fillna(0, inplace=True)

train_scores = []
test_scores = []

for index, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=42)):

    train = data_train.iloc[train_index]
    test = data_train.iloc[test_index]

    labels = train.TARGET.values
    idx = test.ID.values
    answers = test.TARGET.values

    train.drop(['ID', 'TARGET'], axis=1, inplace=True)
    test.drop(['ID', 'TARGET'], axis=1, inplace=True)

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    model, predictions = xgboost_model(train, labels, test)

    train_score = roc_auc_score(labels,
                                model.predict(xgb.DMatrix(train)),
                                average='weighted')
    print('cv train score:', train_score)
    train_scores.append(train_score)

    score = roc_auc_score(answers, predictions, average='weighted')
    print('cv score:', score)
    test_scores.append(score)
    print()

    if submit:
        test_mat = xgb.DMatrix(data_test, missing=9999999999)
        test_predictions = model.predict(test_mat)
        submission.loc[:, 'target_{}'.format(index)] = test_predictions

print()
print('train score:', np.mean(train_scores))
print('cv score:', np.mean(test_scores))

if submit:
    sub = submission.mean(axis=1)
    final_predictions = pd.DataFrame({'ID': test_idx,
                                      'TARGET': sub})
    final_predictions.to_csv('../submissions/{}_v{}.csv'.format('ensemble', version),
                             index=False)
