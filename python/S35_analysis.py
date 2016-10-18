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

df = pd.concat([train, response], axis=1)
df[df["Response"] == 1.0]["S35_C_md5"].value_counts()
df[df["Response"] == 0.0]["S35_C_md5"].value_counts()
categorical = pd.read_hdf("hdf/train_categorical.hdf")

v = [df[df["S35_C_md5"] == idx].index[0]
     for idx in df["S35_C_md5"].value_counts().index]

categorical.loc[v, [c for c in categorical.columns if "S35" in c]]
df.loc[v]
