import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import time
import seaborn as sns
T_FILES = [
    "hdf/train_categorical.hdf",
    "hdf/train_numeric.hdf",
    "hdf/train_date_L0_normalized.hdf",
    "hdf/train_date_L1_normalized.hdf",
    "hdf/train_date_L2_normalized.hdf",
    "hdf/train_date_L3_normalized.hdf",
    "hdf/train_date_min_max.hdf",
    "hdf/train_date_L0_min_max.hdf",
    "hdf/train_date_L1_min_max.hdf",
    "hdf/train_date_L2_min_max.hdf",
    "hdf/train_date_L3_min_max.hdf",
    "hdf/train_magic.hdf",
    "hdf/train_id.hdf",
    "hdf/train_categorical_L3_S32_F3854_decomposite.hdf",
    "hdf/train_numeric_L1_S24_F1844_extract.hdf",
    "hdf/train_numeric_L1_S24_F1723_-0.12.hdf",
    "hdf/train_numeric_L1_S24_F1723_extract.hdf",
    "hdf/train_mask.hdf",
    "hdf/train_S_min_max.hdf",
    "hdf/train_S_C_md5.hdf",
    "hdf/train_S_N_md5.hdf",
]
RESPONSE = "hdf/train_response.hdf"


def select_ccolumns(indices, file_list, remove_columns):
    print("Loading Train Data...")
    X = pd.concat((pd.read_hdf(filename).loc[indices]
                   for filename in file_list), axis=1)
    X.drop(remove_columns, axis=1, inplace=True)
    columns = X.columns
    y = pd.read_hdf(RESPONSE).loc[X.index].values.ravel()
    X = X.values
    # Feature Selection
    print("Selecting Train Features...")
    clf = XGBClassifier(base_score=0.005)
    clf.fit(X, y)
    tmp = [(clf.feature_importances_[i], columns[i])
           for i in range(len(columns))]
    tmp = sorted(tmp, reverse=True)
    columns = [c[1] for c in tmp]
    column_scores = [c[0] for c in tmp]

    return columns, column_scores


def xgboost_bosch(train_indices, train_file_list, important_columns):
    print("Reloading Train Data...")
    X = pd.concat(
        (drop_columns(
            pd.read_hdf(filename).loc[train_indices], important_columns)
         for filename in train_file_list),
        axis=1)
    y = pd.read_hdf(RESPONSE)
    y = y.loc[X.index].values.ravel()
    X = X.values
    # 予測して Cross Validation
    print("Predicting...")
    clf = XGBClassifier(max_depth=5, base_score=0.005)
    cv = StratifiedKFold(y, n_folds=3)
    predictions = np.ones(y.shape[0])
    for i, (train, test) in enumerate(cv):
        predictions[test] = clf.fit(X[train],
                                    y[train]).predict_proba(X[test])[:, 1]
        print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(y[
            test], predictions[test])))
    print(roc_auc_score(y, predictions))
    # MCC 最適化
    thresholds = np.linspace(0.01, 0.99, 200)
    mcc = np.array(
        [matthews_corrcoef(y, predictions > thr) for thr in thresholds])
    plt.plot(thresholds, mcc)
    best_threshold = thresholds[mcc.argmax()]
    print(mcc.max())
    return clf, best_threshold


def drop_columns(df, undrop_columns):
    drop_columns = [c for c in df.columns if c not in undrop_columns]
    df.drop(drop_columns, axis=1, inplace=True)
    return df


def predict(important_columns, test_indices, clf, best_threshold,
            train_file_list):
    test_file_list = []
    for file in train_file_list:
        test_file_list.append(file.replace("train", "test"))
    # テストデータ読み込み
    print("Loading Test Data...")
    X = pd.concat(
        (drop_columns(pd.read_hdf(filename).loc[test_indices], important_columns)
         for filename in test_file_list),
        axis=1)
    X = X.values
    # 0 or 1 に正規化
    predictions = (clf.predict_proba(X)[:, 1] > best_threshold).astype(np.int8)
    # 提出データを生成
    sub = pd.DataFrame(index=test_indices.astype(np.int32), columns=[])
    sub["Response"] = predictions
    return sub

cdf = pd.concat([pd.read_hdf("hdf/train_categorical.hdf"),
                 pd.read_hdf("hdf/train_response.hdf")], axis=1)
tr_a = cdf[cdf["L3_S32_F3851"] == 1.0].index
cdf.drop("L3_S32_F3851", axis=1, inplace=True)
tr_b = np.concatenate([cdf[cdf["Response"] == 1.0].index,
                       cdf[cdf["Response"] == 0.0].sample(n=700000).index])

cdf = pd.read_hdf("hdf/test_categorical.hdf")
te_a = cdf[cdf["L3_S32_F3851"] == 1.0].index
te_b = cdf.drop(te_a).index

del cdf

columns, column_scores = select_ccolumns(
    tr_a, T_FILES, ["S32_C_md5", "S3_C_md5", "S21_C_md5", "S6_C_md5",
                    "S33_C_md5", "S36_C_md5", "S2_C_md5", "S15_C_md5", "S7_C_md5"])

important_columns = [c for c, s in zip(columns, column_scores) if s > 0.005]

clf, best_threshold = xgboost_bosch(tr_a, T_FILES, important_columns)
sub_a = predict(important_columns, te_a, clf, best_threshold, T_FILES)

columns, column_scores = select_ccolumns(
    tr_b, T_FILES, ["S32_C_md5", "S3_C_md5",  "S21_C_md5", "S6_C_md5",
                    "S33_C_md5", "S36_C_md5", "S2_C_md5", "S15_C_md5", "S7_C_md5"])
[(c, s)for c, s in zip(columns, column_scores)]

important_columns = [c for c, s in zip(columns, column_scores) if s > 0.005]
clf, best_threshold = xgboost_bosch(pd.read_hdf(
    "hdf/train_response.hdf").index, T_FILES, important_columns)

sub_b = predict(important_columns, te_b, clf, best_threshold, T_FILES)

sub = pd.concat([sub_a, sub_b])

sub.to_csv("submission.csv.gz", compression="gzip")
