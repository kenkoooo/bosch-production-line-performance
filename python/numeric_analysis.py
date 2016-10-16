import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import time


def twoplot(df, col, xaxis=None):
    ''' scatter plot a feature split into response values as two subgraphs '''
    if col not in df.columns.values:
        print('ERROR: %s not a column' % col)
    ndf = pd.DataFrame(index=df.index)
    ndf[col] = df[col]
    ndf[xaxis] = df[xaxis] if xaxis else df.index
    ndf['Response'] = df['Response']

    g = sns.FacetGrid(ndf, col="Response", hue="Response")
    g.map(plt.scatter, xaxis, col, alpha=.7, s=1)
    g.add_legend()

    del ndf


columns = ['L3_S29_F3351',
           'L3_S30_F3704',
           'L3_S38_F3960',
           'L3_S33_F3865',
           'L3_S33_F3857',
           'L3_S29_F3339',
           'L3_S33_F3859',
           'L3_S30_F3574',
           'L3_S29_F3407',
           'L1_S24_F1846',
           'L1_S24_F1498',
           'L3_S30_F3829',
           'L0_S22_F601',
           'L3_S30_F3794',
           'L3_S29_F3327',
           'L1_S24_F867',
           'L1_S24_F1844',
           'L0_S9_F180',
           'L0_S5_F114',
           'L0_S10_F259',
           'L0_S0_F16',
           'L3_S38_F3956',
           'L3_S30_F3744',
           'L3_S30_F3634',
           'L3_S29_F3436',
           'L2_S26_F3106',
           'L1_S24_F1604',
           'L0_S3_F100',
           'L0_S11_F310',
           'L0_S0_F4',
           'L3_S38_F3952',
           'L3_S36_F3920',
           'L3_S33_F3873',
           'L3_S32_F3850',
           'L3_S29_F3461',
           'L1_S24_F1695',
           'L1_S24_F1632',
           'L0_S4_F104',
           'L3_S30_F3809',
           'L3_S30_F3754',
           'L3_S30_F3749',
           'L3_S30_F3604',
           'L3_S30_F3494',
           'L3_S29_F3479',
           'L3_S29_F3348',
           'L3_S29_F3342',
           'L3_S29_F3336',
           'L3_S29_F3321',
           'L2_S26_F3069',
           'L1_S24_F1763',
           'L1_S24_F1723',
           'L1_S24_F1490',
           'L0_S6_F132',
           'L0_S4_F109',
           'L0_S1_F28',
           'L0_S13_F354',
           'L0_S10_F239',
           'L0_S0_F6',
           'L0_S0_F20']

train = pd.read_hdf("hdf/train_numeric.hdf")
response = pd.read_hdf("hdf/train_response.hdf")
train = pd.concat([train, response], axis=1)

len([c for c in train.columns if "S29_" in c])

for i in range(0, 10):
    for j in range(i + 1, 10):
        twoplot(train, columns[i], columns[j])
