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


train = pd.read_hdf("hdf/train_date_min_max.hdf")
test = pd.read_hdf("hdf/test_date_min_max.hdf")
train_test = pd.concat([train, test])

response = pd.read_hdf("hdf/train_response.hdf")
train_test = pd.concat([train_test, response], axis=1)
twoplot(train_test[train_test["min_max"] < 60], "min_max")

min_max_origin = train_test[["min_max"]]

min_max_30_50 = min_max_origin.where(
    (min_max_origin > 30) & (min_max_origin < 50), 0.0)
min_max_30_50 = min_max_30_50.where(min_max_30_50 == 0.0, 1.0)
min_max_30_50.rename(columns={"min_max": "min_max_30_50"}, inplace=True)
min_max_30_50.loc[train.index].to_hdf(
    "hdf/train_min_max_30_50.hdf", "df", mode="w")
min_max_30_50.loc[test.index].to_hdf(
    "hdf/test_min_max_30_50.hdf", "df", mode="w")

min_max_25 = min_max_origin.where(min_max_origin < 2.5, 0.0)
min_max_25 = min_max_25.where(min_max_25 == 0.0, 1.0)
min_max_25.rename(columns={"min_max": "min_max_25"}, inplace=True)
min_max_25.loc[train.index].to_hdf(
    "hdf/train_min_max_25.hdf", "df", mode="w")
min_max_25.loc[test.index].to_hdf(
    "hdf/test_min_max_25.hdf", "df", mode="w")
