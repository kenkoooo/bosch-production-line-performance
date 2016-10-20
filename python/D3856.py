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


train_date = pd.read_hdf("hdf/train_date.hdf")
test_date = pd.read_hdf("hdf/test_date.hdf")

train_test = pd.concat([train_date, test_date])

response = pd.read_hdf("hdf/train_response.hdf")
train_test = pd.concat([train_test, response], axis=1)

twoplot(train_test.loc[train_date.index], "L3_S33_D3856")

train_test["Id"] = train_test.index
train_test = train_test.sort_values(by=["L3_S33_D3856", "Id"], ascending=True)
train_test['L3_S33_D3856_magic3'] = train_test[
    "Id"].diff().fillna(9999999).astype(int)
train_test['L3_S33_D3856_magic4'] = train_test[
    "Id"].iloc[::-1].diff().fillna(9999999).astype(int)

twoplot(train_test.loc[train_date.index],
        "L3_S33_D3856_magic3", "L3_S33_D3856")
twoplot(train_test.loc[train_date.index], "min")

train_test["min"] = train_test.drop("Response", axis=1).min(axis=1)
train_test["max"] = train_test.drop("Response", axis=1).max(axis=1)
twoplot(train_test.loc[train_date.index], "L3_S33_D3856", "min")

train_test["L3_S33_D3856_normalized"] = train_test[
    "L3_S33_D3856"] - train_test["min"]


twoplot(train_test.loc[train_date.index], "max", 'L3_S33_D3856')

train_test["hoge"] = train_test["L3_S33_D3856_normalized"] * train_test["Id"]

df = pd.DataFrame(index=train_test.index, columns=[])
df["L3_S33_D3856_normalized_cross_Id"] = train_test["hoge"]
