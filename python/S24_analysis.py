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

numeric = pd.concat([pd.read_hdf("hdf/train_numeric.hdf"),
                     pd.read_hdf("hdf/test_numeric.hdf")])
response = pd.read_hdf("hdf/train_response.hdf")

numeric = numeric.loc[:, [c for c in numeric.columns if "S24" in c]]
binary = numeric.where(np.isnan(numeric), 1)
binary["S24_N_binary_md5"] = binary.apply(lambda x: int(
    hashlib.md5(x.values).hexdigest(), 16), axis=1)

df = binary.loc[:, "S24_N_binary_md5"]
df = df.astype(np.float32)

df = pd.concat([df, response], axis=1)

twoplot(df.loc[response.index], "S24_N_binary_md5")

x = df.loc[response.index]
cnt = pd.DataFrame(
    index=x["S24_N_binary_md5"].value_counts().index, columns=[])
cnt["P1"] = x[x["Response"] == 1.0]["S24_N_binary_md5"].value_counts()
cnt["P0"] = x[x["Response"] == 0.0]["S24_N_binary_md5"].value_counts()
cnt["ratio"] = cnt["P1"] / cnt.sum(axis=1)
high_hash = cnt[cnt["P0"] > 5000].sort_values(
    "ratio", ascending=False).index[0:6]

binary = binary.astype(np.float32)
high_idx = [binary[binary["S24_N_binary_md5"] == h].index[0]
            for h in high_hash]
binary.loc[high_idx].to_csv("hoge.csv")
