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

train_categorical = pd.read_hdf("hdf/train_categorical.hdf")
test_categorical = pd.read_hdf("hdf/test_categorical.hdf")


columns = train_categorical.columns

train = pd.read_hdf("hdf/train_response.hdf")
test = pd.DataFrame(index=test_categorical.index, columns=[])

c = "hoge"
X = [x for x in columns if "L3" in x]
train[c] = train_categorical.loc[:, X].min(axis=1)
test[c] = test_categorical.loc[:, X].min(axis=1)

train_test = pd.concat([train, test])
train_test["Id"] = train_test.index

train_test = train_test.sort_values(by=[c, "Id"], ascending=True)
train_test[c + '_magic3'] = train_test["Id"].diff().fillna(9999999).astype(int)
train_test[c + '_magic4'] = train_test["Id"].iloc[::-
                                                  1].diff().fillna(9999999).astype(int)

twoplot(train_test.loc[train.index], c + '_magic3')
twoplot(train_test.loc[train.index], c + '_magic4')
