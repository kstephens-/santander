import pandas as pd
import numpy as np

from scipy import stats

import features as fts


data = pd.read_csv('../data/train.csv')
dqr_f = '../data/dqr.txt'

remove_no_variation = fts.no_variation_features(data)
data.drop(remove_no_variation, axis=1, inplace=True)

remove_duplicated = fts.duplicate_features(data)
data.drop(remove_duplicated, axis=1, inplace=True)

columns = data.columns[1:-1]

with open(dqr_f, 'w') as f:
    for col in columns:

        r = stats.ks_2samp(data.loc[data['TARGET'] == 0, col],
                           data.loc[data['TARGET'] == 1, col])

        print('column:', col, file=f)
        print('num NA:', len(data[data[col].isnull()]), file=f)
        print('num 0:', len(data[data[col] == 0]), file=f)
        print('num unique:', len(pd.unique(data[col])), file=f)
        print('ks statistic:', r.statistic, file=f)
        print('p value:', r.pvalue, file=f)
        print('Column Summary:', file=f)
        print(data[col].describe(), file=f)
        print(file=f)
