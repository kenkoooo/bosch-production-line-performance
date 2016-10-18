import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import time
import hashlib


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


train_all_md5 = pd.read_hdf("hdf/train_all_md5.hdf")
test_all_md5 = pd.read_hdf("hdf/test_all_md5.hdf")

train_test = pd.concat([train_all_md5, test_all_md5])
cnt = train_test["all_md5"].value_counts()

cnt_values = [cnt[value] for value in train_test["all_md5"].values]

train_test["md5_count"] = cnt_values

train_test[["md5_count"]].loc[train_all_md5.index].to_hdf(
    "hdf/train_md5_count.hdf", "df", mode="w")
train_test[["md5_count"]].loc[test_all_md5.index].to_hdf(
    "hdf/test_md5_count.hdf", "df", mode="w")
