import pandas as pd
import numpy as np
import operator
import xgboost as xgb

import features

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer, Binarizer, scale, normalize
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from unbalanced_dataset import UnderSampler, TomekLinks, SMOTE

from scipy.stats.mstats import gmean, hmean

submit = False
version = '1.03'


def xgboost_model(train, labels, test, answers):

    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'auc'

    params['eta'] = 0.01
    params['gamma'] = 0
    params['max_depth'] = 5
    params['min_child_weight'] = 7
    params['max_delta_step'] = 0
    params['subsample'] = 0.8
    params['colsample_bytree'] = 0.5
    params['lambda'] = 1
    params['alpha'] = 0
    params['seed'] = 4242

    # params['eta'] = 0.03
    # params['gamma'] = 0
    # params['max_depth'] = 5
    # params['min_child_weight'] = 1
    # params['max_delta_step'] = 0
    # params['subsample'] = 0.8
    # params['colsample_bytree'] = 0.7
    # params['lambda'] = 1
    # params['alpha'] = 0
    #params['seed'] = 1423

    params['silent'] = 1

    xgtrain = xgb.DMatrix(train, labels, missing=9999999999)
    xgtest = xgb.DMatrix(test, answers, missing=9999999999)
    # xgtrain = xgb.DMatrix(train, labels)
    # xgtest = xgb.DMatrix(test)

    num_rounds = 1500
    m = xgb.train(list(params.items()), xgtrain, num_rounds,
                  evals=[(xgtrain, 'train'), (xgtest, 'eval')],
                  early_stopping_rounds=100, verbose_eval=False)
    return m, m.predict(xgtest)


rfc = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,
    criterion='entropy',
    max_depth=25,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    min_samples_split=150,
    random_state=4
)

rfc = ExtraTreesClassifier(
    n_estimators=600,
    n_jobs=-1,
    criterion='entropy',
    max_depth=35,
    min_samples_split=125,
    min_samples_leaf=1,
    random_state=1
)


if __name__ == '__main__':

    data = pd.read_csv('../data/train.csv')
    y = data['TARGET']

    if submit:
        test_data = pd.read_csv('../data/test.csv')
        test_idx = test_data.ID.values
        submission = pd.DataFrame()

        test_data = features.add_features(test_data)
        test_data.drop(['ID'], axis=1, inplace=True)

        test_data = test_data.fillna(0)
        #test_data = test_data.astype(float)

    train_scores = []
    test_scores = []

    drop_no_variation = features.no_variation_features(data)
    data.drop(drop_no_variation, axis=1, inplace=True)
    # test.drop(drop_no_variation, axis=1, inplace=True)

    drop_duplicates = features.duplicate_features(data)
    data.drop(drop_duplicates, axis=1, inplace=True)
    # test.drop(drop_duplicates, axis=1, inplace=True)


    for index, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=42)):

        train = data.iloc[train_index]
        test = data.iloc[test_index]

        #train, test = features.add_train_dependant_features(train, test)

        #train.drop_duplicates(train.columns.difference(['ID']), inplace=True)

        labels = train.TARGET.values
        idx = test.ID.values
        answers = test.TARGET.values

        # sampler = TomekLinks()
        # train, labels = sampler.fit_transform(train, labels)
        # test = sampler.transform(test)
        # use_features = train.columns[1: -1]
        # train.loc[:, 'SumZeros'] = (train[use_features] == 0).astype(int).sum(axis=1)
        # test.loc[:, 'SumZeros'] = (test[use_features] == 0).astype(int).sum(axis=1)

        # add features
        train = features.add_features(train)
        test = features.add_features(test)

        # drop_no_variation = features.no_variation_features(train)
        # train.drop(drop_no_variation, axis=1, inplace=True)
        # test.drop(drop_no_variation, axis=1, inplace=True)

        # drop_duplicates = features.duplicate_features(train)
        # train.drop(drop_duplicates, axis=1, inplace=True)
        # test.drop(drop_duplicates, axis=1, inplace=True)

        #train = features.add_default_feature_count(train)
        #test = features.add_default_feature_count(test)
        #train = features.zero_count(train)
        #test = features.zero_count(test)
        #train, test = features.add_default_feature_count(train, test)
        #train, test = features.var15_dummy(train, test)

        train.drop(['ID', 'TARGET'], axis=1, inplace=True)
        test.drop(['ID', 'TARGET'], axis=1, inplace=True)

        #train, test = features.add_default_feature_count(train, test)

        train = train.fillna(0)
        #train = train.astype(float)

        test = test.fillna(0)
        #test = test.astype(float)

        pca = PCA(n_components=2)
        x_train_projected = pca.fit_transform(normalize(train, axis=0))
        x_test_projected = pca.transform(normalize(test, axis=0))

        train.loc[:, 'PCAOne'] = x_train_projected[:, 0]
        train.loc[:, 'PCATwo'] = x_train_projected[:, 1]
        # train.loc[:, 'PCAThree'] = x_train_projected[:, 2]
        # train.loc[:, 'PCAFour'] = x_train_projected[:, 3]
        # train.loc[:, 'PCAFive'] = x_train_projected[:, 4]
        # train.loc[:, 'PCASix'] = x_train_projected[:, 5]
        #train.loc[:, 'PCASeven'] = x_train_projected[:, 6]
        # train.loc[:, 'PCAEight'] = x_train_projected[:, 7]
        # train.loc[:, 'PCANine'] = x_train_projected[:, 8]
        # train.loc[:, 'PCATen'] = x_train_projected[:, 9]
        # train.loc[:, 'PCAEleven'] = x_train_projected[:, 10]
        # train.loc[:, 'PCATwelve'] = x_train_projected[:, 11]
        # train.loc[:, 'PCAThirtenn'] = x_train_projected[:, 12]

        test.loc[:, 'PCAOne'] = x_test_projected[:, 0]
        test.loc[:, 'PCATwo'] = x_test_projected[:, 1]
        # test.loc[:, 'PCAThree'] = x_test_projected[:, 2]
        # test.loc[:, 'PCAFour'] = x_test_projected[:, 3]
        # test.loc[:, 'PCAFive'] = x_test_projected[:, 4]
        # test.loc[:, 'PCASix'] = x_test_projected[:, 5]
        #test.loc[:, 'PCASeven'] = x_test_projected[:, 6]
        # test.loc[:, 'PCAEight'] = x_test_projected[:, 7]
        # test.loc[:, 'PCANine'] = x_test_projected[:, 8]
        # test.loc[:, 'PCATen'] = x_test_projected[:, 9]
        # test.loc[:, 'PCAEleven'] = x_test_projected[:, 10]
        # test.loc[:, 'PCATwelve'] = x_test_projected[:, 11]
        # test.loc[:, 'PCAThirtenn'] = x_test_projected[:, 12]

        # imp = Imputer(strategy='mean')
        # train = imp.fit_transform(train, labels)
        # test = imp.transform(test)
        train_bin = Binarizer().fit_transform(scale(train))
        select_chi2 = SelectPercentile(chi2, percentile=95).fit(train_bin, labels)
        select_f_classif = SelectPercentile(f_classif, percentile=95).fit(scale(train), labels)

        chi2_selected = select_chi2.get_support()
        f_classif_selected = select_f_classif.get_support()
        selected = chi2_selected & f_classif_selected
        sel_features = [f for f,s in zip(train.columns, selected) if s]

        train = train[sel_features]
        test = test[sel_features]

        print()
        print(train.shape)
        print(test.shape)

        model, predictions = xgboost_model(train, labels, test, answers)
        #rfc.fit(train, labels)

        cols = train.columns
        feature_importance = model.get_fscore()
        norm_feats = {int(f.strip('f')): feature_importance[f]/sum(feature_importance.values())
            for f in feature_importance}

        for f, imp in sorted(norm_feats.items(), key=operator.itemgetter(1), reverse=True)[:50]:
            print('{}\t{}'.format(cols[f], imp))
        print()

        train_score = roc_auc_score(labels,
                                    model.predict(xgb.DMatrix(train)),
                                    average='weighted')
        # train_score = roc_auc_score(labels,
        #                             rfc.predict_proba(train)[:, 1],
        #                             average='weighted')
        # predictions = rfc.predict_proba(test)[:, 1]

        print('cv train score:', train_score)
        train_scores.append(train_score)

        score = roc_auc_score(answers, predictions, average='weighted')
        print('cv score:', score)
        test_scores.append(score)

        if submit:
            x_test_data_projected = pca.transform(normalize(test_data, axis=0))
            test_sel = test_data[sel_features]
            test_sel.loc[:, 'PCAOne'] = x_test_data_projected[:, 0]
            test_sel.loc[:, 'PCATwo'] = x_test_data_projected[:, 1]

            test_mat = xgb.DMatrix(test_sel, missing=9999999999)
            test_predictions = model.predict(test_mat)
            submission.loc[:, 'target_{}'.format(index)] = test_predictions

    print()
    print('train score:', np.mean(train_scores))
    print('cv score:', np.mean(test_scores))
    # print('train score:', hmean(train_scores))
    # print('cv score:', hmean(test_scores))
    print()

    if submit:
        sub = submission.mean(axis=1)
        final_predictions = pd.DataFrame({'ID': test_idx,
                                          'TARGET': sub})
        final_predictions.to_csv('../submissions/{}_v{}.csv'.format('xgb', version),
                                 index=False)
