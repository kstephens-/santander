import pandas as pd
import numpy as np

from scipy.spatial import distance
from sklearn.preprocessing import maxabs_scale


def no_variation_features(train):

    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)

    return remove


def duplicate_features(train):

    remove = []
    c = train.columns
    for i in range(len(c)-1):
        v = train[c[i]].values
        for j in range(i+1, len(c)):
            if np.array_equal(v, train[c[j]].values):
                remove.append(c[j])
    return remove


def remove_empty_columns(df):

    df.drop(['ind_var2_0', 'ind_var2', 'ind_var27_0', 'ind_var28_0',
             'ind_var28', 'ind_var27', 'ind_var41', 'ind_var46_0',
             'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28',
             'num_var27', 'num_var41', 'num_var46_0', 'num_var46',
             'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46',
             'delta_imp_reemb_var33_1y3', 'delta_imp_trasp_var17_out_1y3',
             'delta_num_reemb_var33_1y3', 'delta_num_trasp_var17_out_1y3',
             'imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3',
             'imp_reemb_var17_hace3',
             'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3',
             'imp_trasp_var17_out_ult1', 'imp_trasp_var33_out_hace3',
             'num_var2_0_ult1', 'num_var2_ult1',
             'num_var2_ult1', 'num_reemb_var13_hace3',
             'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3',
             'num_trasp_var33_out_hace3', 'saldo_var2_ult1',
             'saldo_medio_var13_medio_hace3',
             'imp_reemb_var33_ult1', 'num_reemb_var17_hace3',
             'num_reemb_var33_ult1', 'num_trasp_var17_out_ult1',
             'saldo_medio_var29_hace3'], axis=1, inplace=True)

    return df


def remove_highly_correlated_columns(train, test):

    t = train.ix[:, train.columns.difference(['TARGET'])]

    train_test = pd.concat([t, test], axis=0)

    tt = train_test.ix[:, train_test.columns.difference(['ID'])]
    c = tt.corr()

    remove = set()
    checked = set()
    for col in c.columns:
        dupes = c.loc[
            (c[col] >= 0.99) & (pd.notnull(c[col])), col].index
        col_not_0 = len(tt[tt[col] != 0])
        for dupe in dupes:
            if dupe == col:
                continue
            if not dupe in remove and dupe not in checked:
                if len(tt[tt[dupe] !=0]) > col_not_0:
                    remove.add(dupe)
                else:
                    remove.add(col)
                # if col_not_0 < len(tt[tt[dupe] != 0]):
                #     remove.add(col)
                # else:
                #     remove.add(dupe)
            checked.add(col)
    train.drop(list(remove), axis=1, inplace=True)
    test.drop(list(remove), axis=1, inplace=True)
    return train, test


def remove_highly_correlated_columns_man(df):

    df.drop(['ind_var13_medio', 'num_var29_0', 'ind_var18', 'ind_var39',
             'ind_var37', 'ind_var34', 'ind_var32', 'num_var32',
             'num_var34', 'num_var37', 'num_var39', 'num_var13_medio',
             'saldo_var29', 'delta_num_reemb_var13_1y3',
             'saldo_medio_var13_medio_ult1', 'ind_var29_0',
             'delta_num_trasp_var33_in_1y3', 'delta_num_trasp_var17_in_1y3',
             'num_var18', 'ind_var25', 'ind_var26', 'ind_var29',
             'num_var29', 'num_var26', 'num_var25', 'delta_num_trasp_var33_out_1y3',
             'delta_num_reemb_var17_1y3'], axis=1, inplace=True)

    # remove highly correlated columns
    # df.drop(['ind_var29_0', 'num_var29_0', 'num_var6',
    #          'num_var13_corto', 'num_var13_medio_0', 'num_meses_var13_medio_ult3',
    #          'num_var18_0', 'delta_imp_amort_var18_1y3', 'ind_var26',
    #          'ind_var32', 'num_var34_0', 'delta_imp_amort_var34_1y3',
    #          'ind_var39', 'num_var26', 'num_var32', 'saldo_var29',
    #          'imp_amort_var18_ult1', 'delta_num_aport_var17_1y3',
    #          'delta_num_compra_var44_1y3', 'num_reemb_var13_ult1',
    #          'num_trasp_var17_in_ult1',
    #          'delta_num_trasp_var33_out_1y3', 'num_trasp_var33_out_ult1',
    #          'num_var7_emit_ult1', 'num_var6_0',
    #          'ind_var29', 'num_var29', 'ind_var13_medio', 'num_var13_medio',
    #          'ind_var18', 'num_var18', 'num_var24', 'ind_var25', 'ind_var34',
    #          'num_var34', 'ind_var37', 'num_var44', 'num_var25', 'num_var37',
    #          'saldo_medio_var13_medio_ult1', 'delta_num_aport_var13_1y3',
    #          'delta_num_aport_var33_1y3', 'delta_num_reemb_var13_1y3',
    #          'delta_num_reemb_var17_1y3',
    #          'delta_num_trasp_var17_in_1y3',
    #          'delta_num_trasp_var33_in_1y3', 'imp_trasp_var33_out_ult1',
    #          'delta_num_venta_var44_1y3', 'num_trasp_var17_in_hace3',
    #          'num_op_var39_efect_ult3'], axis=1, inplace=True)

    # remove linear combination columns
    # df.drop(['ind_var13', 'num_var1', 'num_var8', 'num_var14_0',
    #          'num_var17_0', 'num_var20_0', 'num_var31_0', 'num_var40_0',
    #          'num_var39', 'saldo_var1', 'saldo_var13', 'ind_var13_0',
    #          'num_var1_0', 'num_var8_0', 'num_var13_0', 'num_var14',
    #          'num_var20', 'num_op_var39_hace2', 'num_var31', 'num_var33',
    #          'num_var40', 'saldo_var12', 'num_var22_ult3', 'num_var45_ult3',
    #          'saldo_var40'], axis=1, inplace=True)
    return df


def remove_non_distinct_columns(df):

    df.drop(['var3'], axis=1, inplace=True)

    return df


def var3_999999(df):

    df.loc[df['var3'] == -999999, 'var3_na'] = 1
    df.loc[df['var3'] == -999999, 'var3'] = 2

    # df.loc[:, 'saldo_medio_var5_hace3_var38_minus'] = \
    #     df['var38'] - df['saldo_medio_var5_hace3']

    # df.loc[:, 'saldo_medio_var5_hace3_var38'] = \
    #     df['var38'] + df['saldo_medio_var5_hace3']

    #df.loc[df['var38'] == 117310.979016, 'var38_default'] = 1

    #df.replace(9999999999, 1.0)
    #df.dropna(inplace=True)

    #df.loc[df['var36'] == 99, 'var36_na'] = 1
    # df.loc[df['var36'] == 99, 'var36'] = 1

    #df.loc[:, 'log_var15_var3'] = np.log1p(df['var15']) + np.log1p(df['var3'])
    df.loc[:, 'log_var15_var3'] = np.log(df['var15']) + np.log(df['var3'])
    df.loc[df['log_var15_var3'] == float('-inf'), 'log_var15_var3'] = 0

    # df.loc[:, 'imp_op_var39_ult1_ult3'] = \
    #     (df['imp_op_var39_comer_ult1'] == df['imp_op_var39_comer_ult3']).astype(float)

    #df.loc[(df['num_meses_var5_ult3'] == 1) & (df['var36'] == 1), 'num_meses_var5_num_var36'] = 1
    # df.loc[(df['num_meses_var5_ult3'] == 0) & (df['num_var45_ult3'] > 300),
    #            'high_num_var45_ult3'] = 1

    # df.loc[:, 'ind_var1_a_0'] = \
    #     df['ind_var1'] - df['ind_var1_0']

    # mx = MaxAbsScaler()
    # dx = mx.fit_transform(df[['var15', 'var38']])
    # df.loc[:, 'var15_var38'] = dx[:, 1] / dx[:, 0]

    return df


def var36_factor(df):

    df.loc[df['var36'] == 0, 'var36_0'] = 1
    #df.loc[df['var36'] == 99, 'var36_na'] = 1
    #df.loc[df['var36'] == 99, 'var36'] = 0

    #df1 = df[['ID', 'var36']]
    #df1.loc[df1['var36'] == 0, 'var36_0'] = 1
    #d = pd.get_dummies(df['var36'])
    #df = pd.concat([df, d], axis=1)

    #df.drop(['var36'], axis=1, inplace=True)

    # print()
    # print('df1 counts')
    # print(df1['var36'].value_counts())
    # print()
    # print('d columns')
    # print(d.columns)
    # print()

    #df2 = pd.concat([df1, d], axis=1)
    #df2.drop(['var36'], axis=1, inplace=True)

    #return pd.merge(df, df2, on='ID', how='left')
    #df.loc[(df['var21'] <= 4500) & (df['var21'] > 0), 'var21_high'] = 1

    # probability of 1 given var 36 == 99


    return df


def var_rel(df):

    #df.loc[(df['var36'] == 2), 'var_36_15'] = df['var15'
    df.loc[:, 'imp_op_var39'] =  \
        ((df['imp_op_var40_comer_ult1'] == df['imp_op_var40_comer_ult3']) & \
        (df['imp_op_var40_comer_ult3'] == df['imp_op_var40_ult1']) & \
        (df['imp_op_var40_comer_ult1'] == df['imp_op_var40_ult1']) & \
        (df['imp_op_var40_comer_ult1'] != 0)).astype(float)
    return df


def clean_data(df):

    df1 = var3_999999(df)
    #df2 = var36_factor(df)
    # df2.drop(['var36'], axis=1, inplace=True)

    return df1


def add_features(df):

    #df = remove_empty_columns(df)
    #test = remove_empty_columns(test)

    #train, test = remove_highly_correlated_columns(train, test)
    #df = remove_highly_correlated_columns_man(df)
    #test = remove_highly_correlated_columns_man(test)
    #df = var_rel(df)

    #df = zero_count(df)
    #df = add_default_feature_count(df)
    df = clean_data(df)
    df = zero_count(df)

    #df = add_default_feature_count(df)
    #test = clean_data(test)

    return df


def add_train_dependant_features(train, test):

    # drop = [c for c in train.columns
    #         if c.startswith('delta')]
    # drop.append('ID')

    # test_drop = list(drop)
    # drop.append('TARGET')

    # type_1 = train[train['TARGET'] == 1]
    # type_1.drop(drop, axis=1, inplace=True)
    # type_1.fillna(0, inplace=True)
    # type_1_f = type_1.astype(float)

    # type_1_n = maxabs_scale(type_1_f, axis=0)
    # type_1_mu = np.mean(type_1_n, axis=0)

    # mu_1 = np.reshape(type_1_mu, (1, -1))

    # train_d = train.drop(drop, axis=1)
    # test_d = test.drop(test_drop, axis=1)

    # train_d.fillna(0, inplace=True)
    # test_d.fillna(0, inplace=True)

    # train_f = train_d.astype(float)
    # test_f = test_d.astype(float)

    # train_n = maxabs_scale(train_f, axis=0)
    # test_n = maxabs_scale(test_f, axis=0)
    train_d = train.drop(['ID', 'TARGET'], axis=1)
    test_d = test.drop(['ID'], axis=1)

    mu_1 = train_d[['TARGET'] == 1].mean()
    mu_1 = np.reshape(mu_1.values, (1, -1))

    train_1_sim = distance.cdist(train_d, mu_1, 'euclidean')
    test_1_sim = distance.cdist(test_d, mu_1, 'euclidean')

    train.loc[:, 'type_1_sim'] = train_1_sim
    test.loc[:, 'type_1_sim'] = test_1_sim

    return train, test


def add_pobabilistic_features(train, test):

    count_pos_99 = train.loc[train['TARGET'] == 1, 'var36'].value_counts()[99]
    count_pos_total = train.loc[train['TARGET'] == 1, 'var36'].count()

    count_1_total = train.shape[0]
    count_99_total = train[train['var36'] == 99].shape[0]

    pass


def add_default_feature_count(df):

    df_mode = df[df.columns.difference(['ID', 'TARGET'])].mode(axis=0)
    df_x = df.copy()

    df.loc[:, 'default_feature_count'] = 0

    for c in df_mode.columns:
        df.loc[(df_x[c] == df_mode.loc[0, c]) & (df_x[c] != 0),
               'default_feature_count'] += 1

    return df


def zero_count(df):

    def z_count(x):
        return np.sum(x == 0)

    df.loc[:, 'zero_count'] = df[df.columns.difference(['ID', 'TARGET'])] \
        .apply(z_count, axis=1)
    return df


# def add_default_feature_count(train, test):

#     train_mode = train.mode(axis=0)
#     train_x = train.copy()
#     test_x = test.copy()

#     train.loc[:, 'default_feature_count'] = 0
#     test.loc[:, 'default_feature_count'] = 0

#     for c in train_mode.columns:

#         train.loc[(train_x[c] == train_mode.loc[0, c]) & (train_x[c] != 0),
#                   'default_feature_count'] += 1
#         test.loc[(test_x[c] == train_mode.loc[0, c]) & (test_x[c] != 0),
#                  'default_feature_count'] += 1

# #     print()
# #     print('default feature count value counts')
# #     print(train['default_feature_count'].value_counts())
# #     print()

#     return train, test

def rare_category(x, category_distribution, cutoff=1, value='Rare'):
    try:
        if category_distribution[x] < cutoff:
            return value
    except (ValueError, KeyError):
        return np.nan
    else:
        return x


def var15_dummy(train, test):
    train_var15 = train[['var15']]
    test_var15 = test[['var15']]

    train_var15.loc[:, 'train'] = True
    test_var15.loc[:, 'train'] = False

    train_distribution = train_var15['var15'].value_counts()

    var15 = pd.concat([train_var15, test_var15])
    var15.loc[:, 'var15'] = var15['var15'] \
        .apply(rare_category, args=(train_distribution, ),
               cutoff=5, value='RareVar15')

    var15_bin = pd.get_dummies(var15['var15'])
    var15_dummy = pd.concat([var15, var15_bin], axis=1)

    msk = var15['train']
    var15_dummy.drop(['train', 'var15', 'RareVar15'], axis=1, inplace=True)

    train_var15s = pd.concat([train, var15_dummy[msk]], axis=1)
    test_var15s = pd.concat([test, var15_dummy[~msk]], axis=1)

    return train_var15s, test_var15s


