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


date = pd.read_hdf("hdf/train_date.hdf")
response = pd.read_hdf("hdf/train_response.hdf")
train = pd.concat([date, response], axis=1)

l0 = date.loc[:, [c for c in date.columns if "L0" in c]]
l1 = date.loc[:, [c for c in date.columns if "L1" in c]]
l2 = date.loc[:, [c for c in date.columns if "L2" in c]]
l3 = date.loc[:, [c for c in date.columns if "L3" in c]]

train["L0_min"] = l0.T.min()
train["L1_min"] = l1.T.min()
train["L2_min"] = l2.T.min()
train["L3_min"] = l3.T.min()

train["L3_min_max"] = l3.T.max() - train["L3_min"]
train["L0_min_max"] = l0.T.max() - train["L0_min"]

twoplot(train, "L0_min_max", "L3_min_max")
