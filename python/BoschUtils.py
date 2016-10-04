import numpy as np
import pandas as pd


def count_columns(gzip_filename):
    chunks = pd.read_csv(gzip_filename, compression="gzip", index_col=0, chunksize=1, dtype=np.float32)
    for chunk in chunks:
        return len(chunk.columns)


def general_df(gz_file):
    return pd.read_csv(gz_file,
                       compression="gzip",
                       index_col=0,
                       dtype=np.float32)


def sampled_data_set(train_files):
    dfs = []
    for train_file in train_files:
        print(train_file)
        df = general_df(train_file)
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def limited_cols_df_values(gz_file, use_cols):
    return pd.read_csv(gz_file, index_col=0, dtype=np.float32, usecols=use_cols).values


def use_cols_list(train_files, indices):
    count_list = []
    for train_file in train_files:
        count_list.append(count_columns(train_file))
    count_list.append(100000)

    cols_list = []
    cur = 0
    for i in range(0, len(count_list) - 1):
        counts = count_list[i]
        l = cur
        cur += counts
        r = cur
        cols = np.concatenate([[0], indices[np.where((l <= indices) & (indices < r))] + 1 - l])
        cols_list.append(cols)
    return cols_list


def entire_data_set(train_files, cols_list):
    values = []
    for train_file, cols in zip(train_files, cols_list):
        print(train_file, cols)
        value = pd.read_csv(train_file,
                            index_col=0,
                            dtype=np.float32,
                            usecols=cols).values
        values.append(value)
    return np.concatenate(values, axis=1)
