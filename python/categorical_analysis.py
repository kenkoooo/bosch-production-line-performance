import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
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
    g.savefig("{}.png".format(col))
    del g
    del ndf


columns = ['L3_S32_C3851',
           'L2_S26_C3038',
           'L1_S25_C1852',
           'L1_S24_C1585',
           'L2_S27_C3131',
           'L2_S27_C3192',
           'L1_S25_C2496',
           'L1_S25_C2779',
           'L1_S24_C695',
           'L1_S25_C2958',
           'L3_S29_C3317',
           'L1_S24_C675',
           'L1_S25_C2229',
           'L0_S10_C215',
           'L2_S26_C3099',
           'L2_S26_C3082',
           'L1_S24_C1137',
           'L1_S25_C2519',
           'L3_S47_C4141',
           'L3_S32_C3853',
           'L1_S24_C1530',
           'L1_S24_C819',
           'L1_S24_C1187',
           'L3_S29_C3475',
           'L1_S24_C1525',
           'L1_S24_C1510',
           'L1_S24_C1278',
           'L0_S9_C154',
           'L0_S2_C35',
           'L1_S24_C1559',
           'L3_S44_C4102',
           'L3_S43_C4061',
           'L2_S28_C3285',
           'L2_S28_C3224',
           'L1_S25_C2099',
           'L1_S25_C2597',
           'L1_S25_C1901',
           'L1_S24_C910',
           'L1_S25_C2802',
           'L1_S24_C710']

train = pd.concat([
    pd.read_hdf("hdf/train_categorical.hdf"),
    pd.read_hdf("hdf/train_response.hdf")], axis=1)
train_test = pd.concat([train, pd.read_hdf("hdf/test_categorical.hdf")])
train_test["Id"] = train_test.index

train_index = train.index
del train

magic_categorical = pd.DataFrame(index=train_test.index, columns=[])
magic_categorical["Response"] = train_test["Response"]

for i in range(len(columns)):
    c = columns[i]
    train_test.sort_values(by=[c, "Id"], ascending=True, inplace=True)
    magic_categorical[
        c + "_magic3"] = train_test["Id"].diff().fillna(9999999).astype(int)
    magic_categorical[
        c + "_magic4"] = train_test["Id"].iloc[::-1].diff().fillna(9999999).astype(int)
    twoplot(magic_categorical.loc[train_index], c + "_magic3")
    twoplot(magic_categorical.loc[train_index], c + "_magic4")
    print(i, len(columns))
magic_categorical.drop("Response", axis=1, inplace=True)
magic_categorical.loc[train_index].to_hdf(
    "hdf/train_magic_categorical.hdf", "df", mode="w")
magic_categorical.drop(train_index).to_hdf(
    "hdf/test_magic_categorical.hdf", "df", mode="w")
