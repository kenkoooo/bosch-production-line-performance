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

train_numeric = pd.read_hdf("hdf/train_numeric.hdf")
test_numeric = pd.read_hdf("hdf/test_numeric.hdf")


columns = train_numeric.columns

train = pd.read_hdf("hdf/train_response.hdf")
test = pd.DataFrame(index=test_numeric.index, columns=[])

i = 0

c = columns[i]
train[c] = train_numeric[c]
test[c] = test_numeric[c]

train_test = pd.concat([train, test])
train_test["Id"] = train_test.index

train_test = train_test.sort_values(by=[c, "Id"], ascending=True)
train_test[c + '_magic3'] = train_test["Id"].diff().fillna(9999999).astype(int)
train_test[c + '_magic4'] = train_test["Id"].iloc[::-
                                                  1].diff().fillna(9999999).astype(int)
twoplot(train_test.loc[train.index], c + '_magic3')
twoplot(train_test.loc[train.index], c + '_magic4')
i += 1
