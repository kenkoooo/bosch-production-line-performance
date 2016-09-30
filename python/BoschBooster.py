import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import time

CHUNK_SIZE = 100000

INPUTS = [
    # "../output/reduced_train_date.csv.gz",
    "../output/date_diff_train.csv.gz",
    "../output/reduced_train_categorical.csv.gz",
    "../output/reduced_train_numeric.csv.gz"
]

TESTS = [
    # "../output/reduced_test_date.csv.gz",
    "../output/date_diff_test.csv.gz",
    "../output/reduced_test_categorical.csv.gz",
    "../output/reduced_test_numeric.csv.gz"
]

RESPONSE = "../output/train_response.csv.gz"


def count_columns(gzip_filename):
    chunks = pd.read_csv(gzip_filename, compression="gzip", index_col=0, chunksize=1, dtype=np.float32)
    for chunk in chunks:
        return len(chunk.columns)


def general_df(gz_file):
    return pd.read_csv(gz_file,
                       compression="gzip",
                       index_col=0,
                       dtype=np.float32)


def general_df_chunk(gz_file):
    return pd.read_csv(gz_file,
                       compression="gzip",
                       index_col=0,
                       chunksize=CHUNK_SIZE,
                       dtype=np.float32)


def sampled_data_set(train_files):
    date_chunks = general_df_chunk(train_files[0])
    categorical_chunks = general_df_chunk(train_files[1])
    num_chunks = general_df_chunk(train_files[2])
    X = pd.concat(
        [pd.concat([d, c, n], axis=1) for d, c, n in zip(date_chunks, categorical_chunks, num_chunks)])
    return X


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


def main():
    print("Loading Sampled Data Set ...")
    X = sampled_data_set(INPUTS)
    y = general_df(RESPONSE).loc[X.index].values.ravel()
    X = X.values

    print("XGBoost in Sampled Data Set ...")
    clf = XGBClassifier(base_score=0.005)
    clf.fit(X, y)

    important_indices = np.where(clf.feature_importances_ > 0.005)[0]
    print(important_indices)

    cols_list = use_cols_list(INPUTS, important_indices)

    print("Loading Entire Data Set ...")
    X = entire_data_set(INPUTS, cols_list)
    y = general_df(RESPONSE).values.ravel()

    print("XGBoost in Entire Data Set ...")
    clf = XGBClassifier(max_depth=5, base_score=0.005)
    cv = StratifiedKFold(y, n_folds=3)

    print("Training in Entire Data Set ...")
    predictions = np.ones(y.shape[0])
    for i, (train, test) in enumerate(cv):
        predictions[test] = clf.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(y[test], predictions[test])))
    print(roc_auc_score(y, predictions))

    print("Picking the best threshold out-of-fold ...")
    thresholds = np.linspace(0.01, 0.99, 50)
    mcc = np.array([matthews_corrcoef(y, predictions > thr) for thr in thresholds])
    best_threshold = thresholds[mcc.argmax()]
    print(mcc.max())

    print("Loading Test Data ...")
    X = entire_data_set(TESTS, cols_list)

    print("Predict in Test Data ...")
    predictions = (clf.predict_proba(X)[:, 1] > best_threshold).astype(np.int8)

    print("Generating submission.csv.gz ...")
    sub = pd.read_csv("../input/sample_submission.csv", index_col=0)
    sub["Response"] = predictions
    sub.to_csv("submission.csv.gz", compression="gzip")

    print("Done!")


if __name__ == '__main__':
    main()
