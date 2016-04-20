import pandas as pd
import numpy as np
import xgboost as xgb
import operator

import features as fts

from scipy.sparse import csr_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer, Binarizer, scale, normalize
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.decomposition import PCA


# imp_op_var40_comer_ult3 9.953338747949612e-06
# num_var40_0   9.953338747949612e-06
# ind_var13_corto_0   5.972003248769767e-06
# num_var31   3.981335499179845e-06
# num_var24_0   3.981335499179845e-06
# ind_var8   3.981335499179845e-06
# num_var12   1.9906677495899225e-06
# num_var14   1.9906677495899225e-06
# ind_var20_0   1.9906677495899225e-06
# num_op_var40_ult3   1.9906677495899225e-06

lowest_ranking_features = [
    'ind_var12',
    'num_var31_0',
    'num_op_var40_comer_ult3',
    'ind_var31',
    'num_var31',
    'num_var8',
    'imp_var7_recib_ult1',
    'ind_var24_0',
    'ind_var24',
    'num_var13_corto',
    'ind_var31_0'
]



# lowest_ranking_features = [
#     'ind_var9_ult1',                   # 9.448056357656174e-05
#     'saldo_medio_var12_hace3',         # 9.448056357656174e-05
#     'num_var5',                        # 9.448056357656174e-05
#     'ind_var5_0',                      # 9.448056357656174e-05
#     'num_var13',                       # 8.267049312949151e-05
#     'num_var12_0',                     # 8.267049312949151e-05
#     'ind_var13',                       # 7.676545790595641e-05
#     'ind_var12_0',                     # 7.676545790595641e-05
#     'num_var25',                       # 7.676545790595641e-05
#     'num_var14_0',                     # 7.676545790595641e-05
#     'delta_imp_aport_var13_1y3',       # 7.08604226824213e-05
#     'num_var13_0',                     # 5.905035223535108e-05
#     'ind_var37_0',                     # 5.905035223535108e-05
#     'ind_var13_0',                     # 4.724028178828087e-05
#     'num_aport_var13_hace3',           # 4.1335246564745756e-05
#     'ind_var5',                        # 3.543021134121065e-05
#     'ind_var26',                       # 2.3620140894140434e-05
#     'num_op_var40_comer_ult3',         # 5.905035223535109e-06
#     'num_op_var40_comer_ult1',         # 5.905035223535109e-06
#     'ind_var25_cte',                   # 9.654428259079422e-05
#     'ind_var14_0',                     # 7.382798080472499e-05
#     'num_var25_0',                     # 7.382798080472499e-05
#     'num_meses_var12_ult3',            # 6.814890535820769e-05
#     'delta_num_aport_var13_1y3',       # 5.111167901865576e-05
#     'num_trasp_var11_ult1',            # 5.111167901865576e-05
#     'num_var40_0',                     # 3.975352812562115e-05
#     'saldo_var14',                     # 3.407445267910384e-05
#     'ind_var26_0',                     # 1.703722633955192e-05
#     'num_aport_var13_ult1',            # 1.1358150893034614e-05
#     'imp_op_var40_ult1',               # 1.1358150893034614e-05
#     'imp_op_var40_comer_ult1',         # 5.679075446517307e-06
#     'saldo_var40',                     # 5.679075446517307e-06
#     'num_var14',                       # 5.905035223535109e-06
#     'ind_var31',                       # 5.905035223535109e-06
#     'num_op_var40_ult1',               # 5.905035223535109e-06
#     'num_var31_0',                     # 5.905035223535109e-06
#     'saldo_medio_var13_corto_hace3',   # 5.905035223535109e-06
#     'num_var12',                       # 5.884187423137801e-06
#     'num_var13_corto_0',               # 5.884187423137801e-06
#     'ind_var13_corto_0',               # 5.900749395173187e-06
#     'ind_var8',                        # 5.900749395173187e-06
#     'ind_var40_0',                     # 5.900749395173187e-06
#     'ind_var25',                       # 5.900749395173187e-06
#     'saldo_var1',                      # 5.900749395173187e-06
#     'ind_var31_0',                     # 6.064354934565611e-06
#     'saldo_var31',                     # 6.064354934565611e-06
#     'imp_op_var40_comer_ult3',         # 6.064354934565611e-06
#     'ind_var12',                       # 6.064354934565611e-06
#     'ind_var25_0',                     # 6.161125760899031e-06
#     'num_var31',                       # 6.161125760899031e-06
#     'imp_var7_recib_ult1',             # 6.161125760899031e-06
#     'num_var8',                        # 6.161125760899031e-06
#     'num_var24',                       # 6.161125760899031e-06
#     'num_var24_0',                     # 6.161125760899031e-06
#     'ind_var24',                       # 6.161125760899031e-06
#     'ind_var24_0',                     # 6.161125760899031e-06
#     'ind_var1_0',                      # 6.161125760899031e-06
#     'num_var1_0'                       # 6.161125760899031e-06
# ]

# lowest_ranking_features = [
#     'ind_var31_0',                     # 6.064354934565611e-06
#     'saldo_var31',                     # 6.064354934565611e-06
#     'imp_op_var40_comer_ult3',         # 6.064354934565611e-06
#     'ind_var12',                       # 6.064354934565611e-06
#     'ind_var25_0',                     # 6.161125760899031e-06
#     'num_var31',                       # 6.161125760899031e-06
#     'imp_var7_recib_ult1',             # 6.161125760899031e-06
#     'num_var8',                        # 6.161125760899031e-06
#     'num_var24',                       # 6.161125760899031e-06
#     'num_var24_0',                     # 6.161125760899031e-06
#     'ind_var24',                       # 6.161125760899031e-06
#     'ind_var24_0',                     # 6.161125760899031e-06
#     'ind_var1_0',                      # 6.161125760899031e-06
#     'num_var1_0'                       # 6.161125760899031e-06
# ]

# lowest_ranking_features = [
#     'ind_var24', #  6.166064447705608e-06
#     'ind_var25_0',   #   6.166064447705608e-06
#     'imp_op_var40_comer_ult3', #  6.166064447705608e-06
#     'ind_var8',   #  6.166064447705608e-06
#     'imp_op_var40_ult1',   #  6.166064447705608e-06
#     'num_op_var40_ult3',  #   6.166064447705608e-06
#     'saldo_var31',   #   6.166064447705608e-06
#     'num_aport_var13_ult1', #  6.166064447705608e-06
#     'ind_var24_0', #  6.166064447705608e-06
#     'num_op_var40_ult1',  # 6.166064447705608e-06
#     'num_var13_corto_0',  #  6.166064447705608e-06
# ]

# tokeep = ['PCAThree',
#           'PCAFive',
#           'PCASix',
#           'PCASeven',
#           'PCAEight',
#           'PCANine',
#           'log_var15_var3',
#           'num_var39_0',  # 0.00031104199066874026
#           'ind_var13',  # 0.00031104199066874026
#           'num_op_var41_comer_ult3',  # 0.00031104199066874026
#           'num_var43_recib_ult1',  # 0.00031104199066874026
#           'imp_op_var41_comer_ult3',  # 0.00031104199066874026
#           'num_var8',  # 0.00031104199066874026
#           'num_var42',  # 0.00031104199066874026
#           'num_var30',  # 0.00031104199066874026
#           'saldo_var8',  # 0.00031104199066874026
#           'num_op_var39_efect_ult3',  # 0.00031104199066874026
#           'num_op_var39_comer_ult3',  # 0.00031104199066874026
#           'num_var41_0',  # 0.0006220839813374805
#           'num_op_var39_ult3',  # 0.0006220839813374805
#           'saldo_var13',  # 0.0009331259720062209
#           'num_var30_0',  # 0.0009331259720062209
#           'ind_var37_cte',  # 0.0009331259720062209
#           'ind_var39_0',  # 0.001244167962674961
#           'num_var5',  # 0.0015552099533437014
#           'ind_var10_ult1',  # 0.0015552099533437014
#           'num_op_var39_hace2',  # 0.0018662519440124418
#           'num_var22_hace2',  # 0.0018662519440124418
#           'num_var35',  # 0.0018662519440124418
#           'ind_var30',  # 0.0018662519440124418
#           'num_med_var22_ult3',  # 0.002177293934681182
#           'imp_op_var41_efect_ult1',  # 0.002488335925349922
#           'var36',  # 0.0027993779160186624
#           'num_med_var45_ult3',  # 0.003110419906687403
#           'imp_op_var39_ult1',  # 0.0037325038880248835
#           'imp_op_var39_comer_ult3',  # 0.0037325038880248835
#           'imp_trans_var37_ult1',  # 0.004043545878693624
#           'num_var5_0',  # 0.004043545878693624
#           'num_var45_ult1',  # 0.004665629860031105
#           'ind_var41_0',  # 0.0052877138413685845
#           'imp_op_var41_ult1',  # 0.0052877138413685845
#           'num_var8_0',  # 0.005598755832037325
#           'imp_op_var41_efect_ult3',  # 0.007153965785381027
#           'num_op_var41_ult3',  # 0.007153965785381027
#           'num_var22_hace3',  # 0.008087091757387248
#           'num_var4',  # 0.008087091757387248
#           'imp_op_var39_comer_ult1',  # 0.008398133748055987
#           'num_var45_ult3',  # 0.008709175738724729
#           'ind_var5',  # 0.009953343701399688
#           'imp_op_var39_efect_ult3',  # 0.009953343701399688
#           'num_meses_var5_ult3',  # 0.009953343701399688
#           'saldo_var42',  # 0.01181959564541213
#           'imp_op_var39_efect_ult1',  # 0.013374805598755831
#           'PCATwo',  # 0.013996889580093312
#           'num_var45_hace2',  # 0.014618973561430793
#           'num_var22_ult1',  # 0.017107309486780714
#           'saldo_medio_var5_ult1',  # 0.017418351477449457
#           'PCAOne',  # 0.018040435458786936
#           'saldo_var5',  # 0.0208398133748056
#           'ind_var8_0',  # 0.021150855365474338
#           'ind_var5_0',  # 0.02177293934681182
#           'num_meses_var39_vig_ult3',  # 0.024572317262830483
#           'saldo_medio_var5_ult3',  # 0.024883359253499222
#           'num_var45_hace3',  # 0.026749611197511663
#           'num_var22_ult3',  # 0.03452566096423017
#           'saldo_medio_var5_hace3',  # 0.04074650077760498
#           'saldo_medio_var5_hace2',  # 0.04292379471228616
#           'SumZeros',  # 0.04696734059097978
#           'saldo_var30',  # 0.09611197511664074
#           'var38',  # 0.1390357698289269
#           'var15']  # 0.20964230171073095

# lowest_ranking_features = [
#     'num_var14', 'imp_op_var40_ult1', 'num_op_var40_ult1',
#     'imp_var7_recib_ult1', 'ind_var31', 'num_var24'
# ]


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# some additional features
features = train.columns[1: -1]
train.insert(1, 'SumZeros', (train[features] == 0).astype(int).sum(axis=1))
test.insert(1, 'SumZeros', (test[features] == 0).astype(int).sum(axis=1))

train.insert(1, 'log_var15_var3', np.log(train['var15']) + np.log(train['var3']))
train.loc[train['log_var15_var3'] == float('-inf'), 'log_var15_var3'] = 0
train['log_var15_var3'].fillna(0.0, inplace=True)

test.insert(1, 'log_var15_var3', np.log(test['var15']) + np.log(test['var3']))
test.loc[test['log_var15_var3'] == float('-inf'), 'log_var15_var3'] = 0
test['log_var15_var3'].fillna(0.0, inplace=True)

# train.insert(1, 'var3_default', (train['var3'] == -999999).astype(int))
# test.insert(1, 'var3_default', (test['var3'] == -999999).astype(int))

# train.loc[train['var3'] == -999999, 'var3'] = 2
# test.loc[test['var3'] == -999999, 'var3'] = 2


#train.insert(1, 'var15_var38_mult', np.sqrt(np.sqrt(train['var15']) + np.sqrt(train['var38'])))
#test.insert(1, 'var15_var38_mult', np.sqrt(np.sqrt(test['var15']) + np.sqrt(test['var38'])))
#train.insert(1, 'var15_var38', train['var38'] * train['var15'])
#test.insert(1, 'var15_var38', test['var38'] * test['var15'])
#train.insert(1, 'var38_saldo_var30', train['var38'] * train['saldo_var30'])
#test.insert(1, 'var38_saldo_var30', test['var38'] * test['saldo_var30'])

# train.insert(1, 'var38_saldo_var30', np.log1p(train['var38']) + np.log1p(train['saldo_var30']))
# train['var38_saldo_var30'].fillna(0.0, inplace=True)
# test.insert(1, 'var38_saldo_var30', np.log1p(test['var38']) + np.log1p(test['saldo_var30']))
# test['var38_saldo_var30'].fillna(0.0, inplace=True)


# remove some features
remove_no_variation = fts.no_variation_features(train)
train.drop(remove_no_variation, axis=1, inplace=True)
test.drop(remove_no_variation, axis=1, inplace=True)

remove_duplicated = fts.duplicate_features(train)
train.drop(remove_duplicated, axis=1, inplace=True)
test.drop(remove_duplicated, axis=1, inplace=True)

features = train.columns[1: -1]
combined_train_test = pd.concat([train[features], test[features]])
normalized_train_test = normalize(combined_train_test[features], axis=0)

pca = PCA(n_components=6)
x_train_test_projected = pca.fit_transform(normalized_train_test)

#x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
#x_test_projected = pca.transform(normalize(test[features], axis=0))

train.insert(1, 'PCAOne', x_train_test_projected[:train.shape[0], 0])
train.insert(1, 'PCATwo', x_train_test_projected[:train.shape[0], 1])
train.insert(1, 'PCAThree', x_train_test_projected[:train.shape[0], 2])
train.insert(1, 'PCAFour', x_train_test_projected[:train.shape[0], 3])
train.insert(1, 'PCAFive', x_train_test_projected[:train.shape[0], 4])
train.insert(1, 'PCASix', x_train_test_projected[:train.shape[0], 5])
#train.insert(1, 'PCASeven', x_train_projected[:, 6])
#train.insert(1, 'PCAEight', x_train_projected[:, 7])
#train.insert(1, 'PCANine', x_train_projected[:, 8])
#train.insert(1, 'PCATen', x_train_projected[:, 9])
#train.insert(1, 'PCAEleven', x_train_projected[:, 10])
#train.insert(1, 'PCATwelve', x_train_projected[:, 11])

test.insert(1, 'PCAOne',  x_train_test_projected[train.shape[0]:, 0])
test.insert(1, 'PCATwo',  x_train_test_projected[train.shape[0]:, 1])
test.insert(1, 'PCAThree', x_train_test_projected[train.shape[0]:, 2])
test.insert(1, 'PCAFour', x_train_test_projected[train.shape[0]:, 3])
test.insert(1, 'PCAFive', x_train_test_projected[train.shape[0]:, 4])
test.insert(1, 'PCASix', x_train_test_projected[train.shape[0]:, 5])
#test.insert(1, 'PCASeven', x_test_projected[:, 6])
#test.insert(1, 'PCAEight', x_test_projected[:, 7])
#test.insert(1, 'PCANine', x_test_projected[:, 8])
#test.insert(1, 'PCATen', x_test_projected[:, 9])
#test.insert(1, 'PCAEleven', x_test_projected[:, 10])
#test.insert(1, 'PCATwelve', x_test_projected[:, 11])

# start dropping low randing features
# features = train.columns[1:-1]
# lowest_ranking_features =  list(set(features)-(set(tokeep)))
#train.drop(lowest_ranking_features, axis=1, inplace=True)
#test.drop(lowest_ranking_features, axis=1, inplace=True)


features = train.columns[1: -1]
repeated_folds = []
for seed in [42, 1039722, 487206]:
    repeated_folds.append(
        StratifiedKFold(train.TARGET.values,
                        n_folds=10,
                        shuffle=True,
                        random_state=seed)
    )
# folds = StratifiedKFold(train.TARGET.values,
#                         n_folds=10,
#                         shuffle=True,
#                         random_state=42)

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'auc'
params['eta'] = 0.01
params['gamma'] = 0
params['max_depth'] = 5
params['min_child_weight'] = 7
params['max_delta_step'] = 0
params['subsample'] = 0.8
params['colsample_bytree'] = 0.7
params['lambda'] = 1
params['alpha'] = 0
params['seed'] = 4242
params['silent'] = 1

# params["eta"] = 0.03
# params["subsample"] = 0.8
# params["colsample_bytree"] = 0.7
# params["silent"] = 1
# params["max_depth"] = 5
# params["min_child_weight"] = 1



num_rounds = 1600

train_train_scores = []
train_test_scores = []

submission = pd.DataFrame({'ID': test.ID.values})
feature_importance = {}
index = 0
for folds in repeated_folds:
    for ind, (train_index, test_index) in enumerate(folds):
        print()
        print('Fold:', index)
        index += 1

        train_train = train.iloc[train_index]
        train_test = train.iloc[test_index]

        print('train shape:', train_train.shape)
        print('test shape:', train_test.shape)

        dtrain_train = xgb.DMatrix(csr_matrix(train_train[features]),
                                   train_train.TARGET.values,
                                   silent=True)
        dtrain_test = xgb.DMatrix(csr_matrix(train_test[features]),
                                  train_test.TARGET.values,
                                  silent=True)

        watchlist = [(dtrain_train, 'train'), (dtrain_test, 'test')]
        model = xgb.train(params, dtrain_train, num_rounds,
                          evals=watchlist, early_stopping_rounds=125,
                          verbose_eval=False)

        train_train_pred = model.predict(dtrain_train, ntree_limit=model.best_iteration)
        train_test_pred = model.predict(dtrain_test, ntree_limit=model.best_iteration)
        train_score = roc_auc_score(train_train.TARGET.values,
                                    train_train_pred,
                                    average='weighted')
        test_score =  roc_auc_score(train_test.TARGET.values,
                                    train_test_pred,
                                    average='weighted')

        dtest = xgb.DMatrix(csr_matrix(test[features]),
                            silent=True)
        test_pred = model.predict(dtest, ntree_limit=model.best_iteration)
        submission.loc[:, 'm_{}'.format(index)] = test_pred

        print('train score:', train_score)
        train_train_scores.append(train_score)
        print('test score:', test_score)
        train_test_scores.append(test_score)

        # feature importance
        cols = train[features].columns
        feature_imp = model.get_fscore()
        for f in feature_imp:
            try:
                feature_importance[cols[int(f.strip('f'))]] += feature_imp[f]
            except KeyError:
                feature_importance[cols[int(f.strip('f'))]] = feature_imp[f]

print()
print('Avg train score:', np.power(np.prod(train_train_scores), 1/index))
print('Avg test score:', np.power(np.prod(train_test_scores), 1/index))
print('Test std:', np.std(train_test_scores))
print()

version = '3.04'
#sub = submission[submission.columns.difference(['ID'])].mean(axis=1)
sub = submission[submission.columns.difference(['ID'])].prod(axis=1)
preds = np.power(sub, 1/index)
final_submission = pd.DataFrame({'ID': test.ID.values,
                                 'TARGET': preds})
final_submission.to_csv('../submissions/{}_v{}.csv'.format('xgb', version),
                        index=False)

imp_total = sum(feature_importance.values())
norm_feats = {col: feature_importance[col]/imp_total for col in feature_importance}
width = max(len(col) for col in norm_feats)
for f, imp in sorted(norm_feats.items(), key=operator.itemgetter(1), reverse=True):
    print('{}\t{}'.format(f.rjust(width), imp))

