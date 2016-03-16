import pandas as pd
import numpy as np
import xgboost as xgb

import features

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score

submit = False
version = '0.03'


def xgboost_model(train, labels, test):

    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'auc'

    params['eta'] = 0.01
    params['gamma'] = 0
    params['max_depth'] = 6
    params['min_child_weight'] = 0.5
    params['max_delta_setp'] = 0
    params['subsample'] = 0.4
    params['colsample_bytree'] = 0.6
    params['lambda'] = 1
    params['alpha'] = 0

    params['silent'] = 1

    xgtrain = xgb.DMatrix(train, labels)
    xgtest = xgb.DMatrix(test)

    num_rounds = 600
    m = xgb.train(list(params.items()), xgtrain, num_rounds)
    return m, m.predict(xgtest)


data = pd.read_csv('../data/train.csv')
y = data['TARGET']

if submit:
    test_data = pd.read_csv('../data/test.csv')
    test_idx = test_data.ID.values
    submission = pd.DataFrame()

    test_data = features.add_features(test_data)
    test_data.drop(['ID'], axis=1, inplace=True)

    test_data = test_data.fillna(0)
    test_data = test_data.astype(float)

    test_mat = xgb.DMatrix(test_data)

train_scores = []
test_scores = []

for index, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=42)):

    train = data.iloc[train_index]
    test = data.iloc[test_index]

    # add features
    train = features.add_features(train)
    test = features.add_features(test)

    #train, test = features.add_train_dependant_features(train, test)

    labels = train.TARGET.values
    idx = test.ID.values
    answers = test.TARGET.values

    train.drop(['ID', 'TARGET'], axis=1, inplace=True)
    test.drop(['ID', 'TARGET'], axis=1, inplace=True)

    train = train.fillna(0)
    train = train.astype(float)

    test = test.fillna(0)
    test = test.astype(float)

    print()
    print(train.shape)
    print(test.shape)

    model, predictions = xgboost_model(train, labels, test)

    train_score = roc_auc_score(labels,
                                model.predict(xgb.DMatrix(train)),
                                average='weighted')
    print('cv train score:', train_score)
    train_scores.append(train_score)

    score = roc_auc_score(answers, predictions, average='weighted')
    print('cv score:', score)
    test_scores.append(score)

    if submit:
        test_predictions = model.predict(test_mat)
        submission.loc[:, 'target_{}'.format(index)] = test_predictions

print()
print('train score:', np.mean(train_scores))
print('cv score:', np.mean(test_scores))
print()

if submit:
    sub = submission.mean(axis=1)
    final_predictions = pd.DataFrame({'ID': test_idx,
                                      'TARGET': sub})
    final_predictions.to_csv('../submissions/{}_v{}.csv'.format('xgb', version),
                             index=False)
