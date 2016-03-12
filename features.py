import pandas as pd
import numpy as np

from sklearn.preprocessing import MaxAbsScaler


def remove_empty_columns(df):

    df.drop(['ind_var2_0', 'ind_var2', 'ind_var27_0', 'ind_var28_0',
             'ind_var28', 'ind_var27', 'ind_var41', 'ind_var46_0',
             'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28',
             'num_var27', 'num_var41', 'num_var46_0', 'num_var46',
             'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46',
             'imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3',
             'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3',
             'num_var2_0_ult1', 'num_var2_ult1', 'num_reemb_var13_hace3',
             'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3',
             'num_trasp_var33_out_hace3', 'saldo_var2_ult1',
             'saldo_medio_var13_medio_hace3',], axis=1, inplace=True)

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

    # remove highly correlated columns
    df.drop(['ind_var29_0', 'num_var29_0', 'num_var6',
             'num_var13_corto', 'num_var13_medio_0', 'num_meses_var13_medio_ult3',
             'num_var18_0', 'delta_imp_amort_var18_1y3', 'ind_var26',
             'ind_var32', 'num_var34_0', 'delta_imp_amort_var34_1y3',
             'ind_var39', 'num_var26', 'num_var32', 'saldo_var29',
             'imp_amort_var18_ult1', 'delta_num_aport_var17_1y3',
             'delta_num_compra_var44_1y3', 'num_reemb_var13_ult1',
             'delta_num_reemb_var33_1y3', 'num_reemb_var33_ult1',
             'num_trasp_var17_in_ult1', 'num_trasp_var17_out_ult1',
             'delta_num_trasp_var33_out_1y3', 'num_trasp_var33_out_ult1',
             'num_reemb_var17_hace3', 'num_var7_emit_ult1', 'num_var6_0',
             'ind_var29', 'num_var29', 'ind_var13_medio', 'num_var13_medio',
             'ind_var18', 'num_var18', 'num_var24', 'ind_var25', 'ind_var34',
             'num_var34', 'ind_var37', 'num_var44', 'num_var25', 'num_var37',
             'saldo_medio_var13_medio_ult1', 'delta_num_aport_var13_1y3',
             'delta_num_aport_var33_1y3', 'delta_num_reemb_var13_1y3',
             'delta_num_reemb_var17_1y3', 'imp_reemb_var33_ult1',
             'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var17_out_1y3',
             'delta_num_trasp_var33_in_1y3', 'imp_trasp_var33_out_ult1',
             'delta_num_venta_var44_1y3', 'num_trasp_var17_in_hace3',
             'num_op_var39_efect_ult3'], axis=1, inplace=True)

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
    df.loc[df['var3'] == -999999, 'var3'] = 0

    # df.loc[df['var36'] == 99, 'var36_na'] = 1
    # df.loc[df['var36'] == 99, 'var36'] = 1

    df.loc[:, 'log_var15_var3'] = np.log(df['var15']) + np.log(df['var3'])

    df.loc[:, 'imp_op_var39_ult1_ult3'] = \
        (df['imp_op_var39_comer_ult1'] == df['imp_op_var39_comer_ult3']).astype(float)

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
    # d = pd.get_dummies(df['var36'])
    # df = pd.concat([df, d], axis=1)

    # print()
    # print('df1 counts')
    # print(df1['var36'].value_counts())
    # print()
    # print('d columns')
    # print(d.columns)
    # print()

    # df2 = pd.concat([df1, d], axis=1)
    # df2.drop(['var36'], axis=1, inplace=True)

    # return pd.merge(df, df2, on='ID', how='left')
    return df


def clean_data(df):

    df1 = var3_999999(df)
    df2 = var36_factor(df)
    # df2.drop(['var36'], axis=1, inplace=True)

    return df2


def add_features(df):

    df = remove_empty_columns(df)
    #test = remove_empty_columns(test)

    #train, test = remove_highly_correlated_columns(train, test)
    df = remove_highly_correlated_columns_man(df)
    #test = remove_highly_correlated_columns_man(test)

    df = clean_data(df)
    #test = clean_data(test)

    return df
