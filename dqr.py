import pandas as pd
import numpy as np

from scipy import stats

import features


data = pd.read_csv('../data/train.csv')
dqr_f = '../data/dqr.txt'

columns = data.columns

data = features.clean_data(data)

with open(dqr_f, 'w') as f:
    for col in columns:

        print('column:', col, file=f)
        print('num NA:', len(data[data[col].isnull()]), file=f)
        print('num 0:', len(data[data[col] == 0]), file=f)
        print('Column Summary:', file=f)
        print(data[col].describe(), file=f)

        r = stats.ks_2samp(data.loc[data['TARGET'] == 0, col],
                           data.loc[data['TARGET'] == 1, col])
        print(file=f)
        print('ks statistic:', r.statistic, file=f)
        print('p value:', r.pvalue, file=f)
        print(file=f)
