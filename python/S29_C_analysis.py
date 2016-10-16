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

train = pd.read_hdf("hdf/train_S_C_md5.hdf")
response = pd.read_hdf("hdf/train_response.hdf")
train = pd.concat([train, response], axis=1)
twoplot(train, "S29_C_md5")
md5_174 = train[train["Response"] == 1.0]["S29_C_md5"].value_counts().index[2]
train[train["S29_C_md5"] == md5_174]

df = pd.read_hdf("hdf/test_S_C_md5.hdf")
df[df["S29_C_md5"] == md5_174]["S29_C_md5"]
ndf = pd.DataFrame(index=df.index, columns=[])
ndf["S29_C_md5_28"] = df[df["S29_C_md5"] == md5_174]["S29_C_md5"]
ndf.where(~np.isnan(ndf), 0.0, inplace=True)
ndf.where(ndf == 0.0, 1.0, inplace=True)
ndf.to_hdf("hdf/test_S29_C_md5_28.hdf", "df", mode="w")
