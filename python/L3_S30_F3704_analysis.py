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


train = pd.read_hdf("hdf/train_numeric.hdf")
response = pd.read_hdf("hdf/train_response.hdf")
train = pd.concat([train, response], axis=1)
len([c for c in train.columns if "L3_S30_F3704" in c])
for c in [c for c in train.columns if "L3_S30_F3704" in c]:
    twoplot(train, c)
