import pandas as pd
import numpy as np
import hashlib
df = pd.read_hdf("hdf/train_S_C_md5.hdf")
response = pd.read_hdf("hdf/train_response.hdf")

categorical = pd.read_hdf("hdf/train_categorical.hdf")
s3_columns = [c for c in categorical.columns if "S3" in c]

s3c = categorical.loc[:, s3_columns]

s3c = pd.concat([s3c, response], axis=1)

s3c[s3c["Response"] == 1.0]

for c in s3c.columns:
    print(c)
    print(s3c[s3c["Response"] == 1.0][c].value_counts())
    print(s3c[s3c["Response"] == 0.0][c].value_counts())
    print("")

f3854_51 = s3c.loc[:, ["L3_S32_F3851", "L3_S32_F3854", "Response"]]

df = f3854_51
df[(df["Response"] == 1.0) & (df["L3_S32_F3851"] == 1.0)][
    "L3_S32_F3854"].value_counts()
df[(df["Response"] == 0.0) & (df["L3_S32_F3851"] == 0.0)][
    "L3_S32_F3854"].value_counts()

df[(np.isnan(df["L3_S32_F3854"])) & (df["Response"] == 1.0)]

df["S21_C_md5"].value_counts()
