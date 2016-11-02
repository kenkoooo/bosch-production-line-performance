import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import time
import seaborn as sns
%matplotlib inline


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
test = pd.read_hdf("hdf/test_numeric.hdf")
train_test = pd.concat([train, test])

response = pd.read_hdf("hdf/train_response.hdf")
train_test = pd.concat([train_test, response], axis=1)
twoplot(train_test, "L3_S29_F3407")

df = pd.DataFrame(index=train_test.index, columns=[])

values = train_test["L3_S29_F3407"].value_counts().index
for value in values:
    tmp = train_test[["L3_S29_F3407"]]
    size = tmp[tmp["L3_S29_F3407"] == value].shape[0]

    tmp2 = train_test[train_test["L3_S29_F3407"] == value]
    ok = tmp2[tmp2["Response"] == 1.0].shape[0]
    ng = tmp2[tmp2["Response"] == 0.0].shape[0]

    ratio = ok / (ok + ng)
    if size < 30000 or ratio < 0.007:
        continue

    tmp = tmp.where(tmp == value, 0.0)
    tmp = tmp.where(tmp == 0.0, 1.0)
    key = "L3_S29_F3407_{v}".format(v=value)
    df[key] = tmp["L3_S29_F3407"]

df.loc[train.index].to_hdf(
    "hdf/train_L3_S29_F3407_decomposite.hdf", "df", mode="w")
df.loc[test.index].to_hdf(
    "hdf/test_L3_S29_F3407_decomposite.hdf", "df", mode="w")
df.columns
