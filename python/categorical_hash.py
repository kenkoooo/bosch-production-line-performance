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


train = pd.read_hdf("hdf/train_S_C_md5.hdf")
test = pd.read_hdf("hdf/test_S_C_md5.hdf")

train_test = pd.concat([train, test])

for i in range(52):
    cnt = train_test["S{}_C_md5".format(i)].value_counts()
    if cnt.size <= 1:
        continue
    p = cnt.values[1] / cnt.sum()
    if p > 0.005:
        print(i)

df = pd.DataFrame(index=train_test.index, columns=[])

for i in [25, 26, 27, 28, 29, 47]:
    column = "S{}_C_md5".format(i)
    cnt = train_test[column].value_counts()
    x = train_test[column]
    for idx, v in zip(cnt.index, cnt.values):
        x[x == idx] = v
    df[column + "_count"] = x
    print(i)
df

response = pd.read_hdf("hdf/train_response.hdf")

x = pd.concat([df, response], axis=1)
twoplot(x.loc[train.index], "S26_C_md5_count")

df.loc[train.index].to_hdf("hdf/train_S_C_md5_count.hdf", "df", mode="w")
df.loc[test.index].to_hdf("hdf/test_S_C_md5_count.hdf", "df", mode="w")
