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

train_date = pd.read_hdf("hdf/train_date.hdf")
test_date = pd.read_hdf("hdf/test_date.hdf")


train = pd.read_hdf("hdf/train_response.hdf")
test = pd.DataFrame(index=test_date.index, columns=[])

storage = pd.concat((train, test))

line = "L0"

train["start_time"] = train_date.loc[
    :, [c for c in train_date.columns if line in c]].min(axis=1)
test["start_time"] = test_date.loc[
    :, [c for c in train_date.columns if line in c]].min(axis=1)

train_test = pd.concat((train, test))
train_test["Id"] = train_test.index

train_test = train_test.sort_values(by=["start_time", "Id"], ascending=True)
train_test[line + '_magic3'] = train_test["Id"].diff().fillna(9999999).astype(int)
train_test[line + '_magic4'] = train_test["Id"].iloc[
    ::-1].diff().fillna(9999999).astype(int)

twoplot(train_test.loc[train.index], line + "_magic3")
twoplot(train_test.loc[train.index], line + "_magic4")
storage[line + '_magic3'] = train_test[line + '_magic3']
storage[line + '_magic4'] = train_test[line + '_magic4']

storage.drop("Response", axis=1, inplace=True)
storage.drop("start_time", axis=1, inplace=True)
storage.loc[train.index].to_hdf("hdf/train_magic_line.hdf", "df", mode="w")
storage.drop(train.index).to_hdf("hdf/test_magic_line.hdf", "df", mode="w")
