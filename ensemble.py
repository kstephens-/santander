import pandas as pd
import numpy as np
import xgboost as xgb

import features

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer, Binarizer, scale, maxabs_scale
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import OneClassSVM

from unbalanced_dataset import UnderSampler, TomekLinks, SMOTE

from scipy.stats.mstats import gmean, hmean

submit = False
version = '2.02'

data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
test_data_idx = test_data['ID'].values
test_data_feat = features.add_features(test_data)
test_data.drop(['ID'], axis=1, inplace=True)

def make_holdout(data):

    holdout_0 = data[data['TARGET'] == 0].sample(frac=0.1)
    holdout_1 = data[data['TARGET'] == 1].sample(frac=0.1)

    holdout = pd.concat([holdout_0, holdout_1])
    holdout_idx = holdout.index
    holdout = holdout.sample(frac=1).reset_index(drop=True)

    new_data = data[~data.index.isin(holdout_idx)]

    return new_data, holdout


#data, holdout = make_holdout(data)
y = data['TARGET']

# holdout_idx = holdout.index.values
# holdout_answers = holdout.TARGET.values
# holdout.drop(['ID', 'TARGET'], axis=1, inplace=True)


train_scores = []
test_scores = []

xgb_classifier = xgb.XGBClassifier(
    missing=9999999999,
    max_depth=5,
    learning_rate=0.01,
    n_estimators=800,
    nthread=4,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=7,
    seed=4242
)

#folds = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=42)
# models = [
#     RandomForestClassifier(n_estimators=300, n_jobs=-1, criterion='gini', random_state=4,
#                            max_depth=20, min_samples_split=100),
#     RandomForestClassifier(n_estimators=150, n_jobs=-1, criterion='entropy', random_state=4,
#                            max_depth=25, min_samples_split=150),
#     ExtraTreesClassifier(n_estimators=150, n_jobs=-1, criterion='gini', random_state=6643),
#     ExtraTreesClassifier(n_estimators=150, n_jobs=-1, criterion='entropy', random_state=11111),
#     xgb_classifier,
#     OneClassSVM(random_state=4928)
# ]

models = [
    RandomForestClassifier(n_estimators=300, n_jobs=-1, criterion='gini', random_state=315,
                           max_depth=20, min_samples_split=100),
    RandomForestClassifier(n_estimators=150, n_jobs=-1, criterion='entropy', random_state=4,
                           max_depth=25, min_samples_split=150),
    xgb_classifier,
    ExtraTreesClassifier(n_estimators=600, n_jobs=-1, criterion='gini', random_state=6643,
                         max_depth=25, min_samples_split=15),
    ExtraTreesClassifier(n_estimators=600, n_jobs=-1, criterion='entropy', random_state=1,
                         max_depth=35, min_samples_split=125)
]

l2_train = pd.DataFrame({'ID': data['ID'], 'TARGET': data['TARGET']})
l2_test = pd.DataFrame({'ID': test_data_idx})
#holdout_test = pd.DataFrame()

cv_seeds = [1, 4232, 151, 9, 35123]

for i, (model, seed) in enumerate(zip(models, cv_seeds)):

    m_train_score = []
    m_test_score = []

    l2_train.loc[:, 'm_{}'.format(i)] = np.zeros(data.shape[0])
    #l2_test.loc[:, 'm_{}'.format(i)] = np.zeros(test.shape[0])

    l2_test_mod = pd.DataFrame()

    folds = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=seed)

    #l2_test = pd.DataFrame()
    #l2_test.loc[:, 'm_{}'.format(i)] = np.zeros(holdout.shape[0])
    #holdout_test.loc[:, 'm_{}'.format(i)] = np.zeros(holdout.shape[0])
    for index, (train_index, test_index) in enumerate(folds):

        train = data.iloc[train_index]
        test = data.iloc[test_index]

        labels = train.TARGET.values
        idx = test.ID.values
        answers = test.TARGET.values

        train_feat = features.add_features(train)
        test_feat = features.add_features(test)
        # test_data_feat = features.add_features(test_data)

        if i == 5:
            one_class_sample = train_feat[train_feat['TARGET'] == 1]

        train_feat.drop(['ID', 'TARGET'], axis=1, inplace=True)
        test_feat.drop(['ID', 'TARGET'], axis=1, inplace=True)
        #test_data_feat.drop(['ID'], axis=1, inplace=True)

        train_feat.fillna(0, inplace=True)
        test_feat.fillna(0, inplace=True)
        test_data_feat.fillna(0, inplace=True)

        train_bin = Binarizer().fit_transform(scale(train))
        select_chi2 = SelectPercentile(chi2, percentile=95).fit(train_bin, labels)
        select_f_classif = SelectPercentile(f_classif, percentile=95).fit(scale(train), labels)

        chi2_selected = select_chi2.get_support()
        f_classif_selected = select_f_classif.get_support()
        selected = chi2_selected & f_classif_selected
        sel_features = [f for f,s in zip(train.columns, selected) if s]

        train_sel = train_feat[sel_features]
        test_sel = test_feat[sel_features]
        test_data_sel = test_data_feat[sel_features]

        if i == 5:
            one_class_sample.fillna(0, inplace=True)
            one_class_sample.drop(['ID', 'TARGET'], axis=1, inplace=True)
            one_class_scaled = maxabs_scale(one_class_sample[sel_features])
            one_class_train = maxabs_scale(train_sel)
            one_class_test = maxabs_scale(test_sel)
            one_class_test_data = maxabs_scale(test_data_sel)

        print()
        print(train_sel.shape)
        print(test_sel.shape)

        if i == 2:
            model.fit(train_sel, labels, eval_metric='auc')
        elif i == 5:
            model.fit(one_class_scaled)
        else:
            model.fit(train_sel, labels)

        if i == 5:
            train_predictions = model.predict(one_class_train)
            predictions = model.predict(one_class_test)
            test_predictions = model.predict(one_class_test_data)

        else:
            train_predictions = model.predict_proba(train_sel)[:,1]
            predictions = model.predict_proba(test_sel)[:,1]
            test_predictions = model.predict_proba(test_data_sel)[:, 1]

        train_score = roc_auc_score(labels, train_predictions, average='weighted')
        test_score = roc_auc_score(answers, predictions, average='weighted')

        print('cv train score:', train_score)
        m_train_score.append(train_score)
        print('cv test score:', test_score)
        m_test_score.append(test_score)
        #holdout_predictions = cls.predict_proba(holdout)[:,1]

        l2_train.loc[test_index, 'm_{}'.format(i)] = predictions
        l2_test_mod.loc[:, 'm_{}'.format(index)] = test_predictions
        #l2_test.loc[:, 'p_{}'.foramt(index)] = holdout_predictions

    #holdout_test.loc[:, 'm_{}'.format(i)] = l2_test.mean(axis=1)
    l2_test_pred = l2_test_mod.mean(axis=1)
    l2_test.loc[:, 'm_{}'.format(i)] = l2_test_pred
    print()
    print('model train score:', np.mean(m_train_score))
    print('model test score:', np.mean(m_test_score))

l2_train.to_csv('../ensemble/train_{}.csv'.format(version), index=False)
l2_test.to_csv('../ensemble/test_{}.csv'.format(version), index=False)

