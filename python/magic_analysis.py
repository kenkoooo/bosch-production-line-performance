import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import time
import hashlib
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


train = pd.concat([
    pd.read_hdf("hdf/train_numeric.hdf"),
    pd.read_hdf("hdf/train_categorical.hdf"),
    pd.read_hdf("hdf/train_magic.hdf"),
    pd.read_hdf("hdf/train_date_min_max.hdf"),
    pd.read_hdf("hdf/train_mask_decomposite.hdf"),
    pd.read_hdf("hdf/train_numeric_pattern_count.hdf")], axis=1)

test = pd.concat([
    pd.read_hdf("hdf/test_numeric.hdf"),
    pd.read_hdf("hdf/test_categorical.hdf"),
    pd.read_hdf("hdf/test_magic.hdf"),
    pd.read_hdf("hdf/test_date_min_max.hdf"),
    pd.read_hdf("hdf/test_mask_decomposite.hdf"),
    pd.read_hdf("hdf/test_numeric_pattern_count.hdf")], axis=1)

train_test = pd.concat([train, test])

response = pd.read_hdf("hdf/train_response.hdf")
train_test = pd.concat([train_test, response], axis=1)

twoplot(train_test[train_test["magic4"] == -1.0], "F3855_magic3")


train_test["Id"] = train_test.index
train_test.sort_values(["L3_S33_F3855", "Id"], inplace=True)
train_test["F3855_magic3"] = train_test["Id"].diff()


train_test[(train_test["L3_S30_F3749"] < -0.1) &
           (train_test["Response"] == 1.0)].shape[0]
train_test[(train_test["L3_S30_F3749"] < -0.1) &
           (train_test["Response"] == 0.0)].shape[0]


a = train_test[(train_test["magic4"] == -1) &
               (train_test["L3_S30_F3749"] < 0.0)].index

df = pd.DataFrame(index=a, columns=[])
df["F3794_m4"] = np.ones(a.shape[0])
train_test["F3794_m4"] = df["F3794_m4"]
df = train_test[["F3794_m4"]]
df = df.fillna(0.0)
df.loc[train.index].to_hdf("hdf/train_F3794_m4.hdf", "df", mode="w")
df.loc[test.index].to_hdf("hdf/test_F3794_m4.hdf", "df", mode="w")

train_test[(train_test["magic4"] == -1)].shape[0]
train_test[(train_test["magic4"] == -1) &
           train_test["Response"] == 1.0].shape[0]
