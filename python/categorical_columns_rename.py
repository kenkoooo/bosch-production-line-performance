import pandas as pd
import numpy as np

df = pd.read_hdf("hdf/test_categorical.hdf")
df.columns = [c.replace("F", "C") for c in df.columns]
df.to_hdf("hdf/test_categorical.hdf", "df", mode="w")
