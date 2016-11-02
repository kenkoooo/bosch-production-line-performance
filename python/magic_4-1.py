import pandas as pd
import numpy as np


train = pd.read_hdf("hdf/train_magic.hdf")
test = pd.read_hdf("hdf/test_magic.hdf")

train_test = pd.concat([train, test])

magic4 = train_test[["magic4"]]
magic4 = magic4.where(magic4 == -1, 0)
magic4 = magic4.where(magic4 != -1, 1)
magic4.rename(columns={"magic4": "magic4-1"}, inplace=True)
magic4_train = magic4.loc[train.index]
magic4_test = magic4.loc[test.index]

magic4_train.to_hdf("hdf/train_magic4-1.hdf", "df", mode="w")
magic4_test.to_hdf("hdf/test_magic4-1.hdf", "df", mode="w")
