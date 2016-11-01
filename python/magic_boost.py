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

T_FILES = [
    # "hdf/train_S_C_md5_count.hdf",
    # "hdf/train_magic_categorical.hdf",
    # "hdf/train_magic_line.hdf",
    # "hdf/train_S24_N_binary_md5.hdf",
    # "hdf/train_date_L0_normalized.hdf",
    # "hdf/train_date_L1_normalized.hdf",
    # "hdf/train_date_L2_normalized.hdf",
    # "hdf/train_date_L3_normalized.hdf",
    # "hdf/train_date_L1_min_max.hdf",
    # "hdf/train_numeric_L1_S24_F1844_extract.hdf",
    # "hdf/train_numeric_L1_S24_F1723_-0.12.hdf",
    # "hdf/train_numeric_L1_S24_F1723_extract.hdf",
    # "hdf/train_S_C_md5.hdf",
    # "hdf/train_S_N_md5.hdf",
    # "hdf/train_S29_C_md5_28.hdf",
]
RESPONSE = "hdf/train_response.hdf"


def select_ccolumns(X):
    print("Loading Train Data...")
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


def xgboost_bosch(train_file_list, important_columns):
    print("Reloading Train Data...")
    X = pd.concat(
        (drop_columns(
            pd.read_hdf(filename), important_columns)
         for filename in train_file_list),
        axis=1)
    y = pd.read_hdf(RESPONSE)
    y = y.loc[X.index].values.ravel()
    column_names = X.columns
    X = X.values
    # 予測して Cross Validation
    print("Predicting...")
    clf = XGBClassifier(max_depth=7, base_score=0.005)
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
    return clf, best_threshold, predictions, column_names


def drop_columns(df, undrop_columns):
    df.drop([c for c in df.columns if c not in undrop_columns],
            axis=1, inplace=True)
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


using_files = (
    [
        'magic4',
        'L0_min',
        'L3_S30_F3749_magic4',
        'L1_S24_F1842',
        'L3_S32_F3854_2',
        'L3_S30_F3704',
        'L3_S29_F3407',
        'L3_S29_F3351',
        'L1_S24_F1844',
        'L3_S38_F3952_magic4',
        'min',
        'L2_min',
        'L1_S24_F1723',
        'L3_S38_F3956_magic4',
        'L1_S24_F1846',
        'min_max',
        'L3_min',
        'L1_S24_F1498',
        'L0_S11_F322',
        'L3_S33_F3857_magic4',
        'L3_S32_F3854_4',
        'L3_S30_F3744_magic3',
        'L3_S29_F3461_magic4',
        'L1_S24_F867',
        'L0_S9_F180',
        'mask',
        'S33_min',
        'S32_min',
        'L3_S33_F3865',
        'L3_S32_F3850_magic4',
        'L3_S30_F3829_magic3',
        'max',
        'L3_S38_F3960_magic4',
        'L3_S38_F3960',
        'L3_S33_F3873',
        'L3_S33_F3859',
        'L3_S30_F3754_magic3',
        'L3_S29_F3461',
        'L3_S29_F3339',
        'L2_S26_F3040',
        'L1_S24_F1723_magic3',
        'L1_S24_F1632',
        'L0_S4_F104',
        'L3_max',
        'L3_S33_F3873_magic3',
        'L3_S33_F3857',
        'L3_S32_F3854_48',
        'L3_S30_F3749',
        'L1_S24_F1758',
        'L1_S24_F1494',
        'L0_S21_F497',
        'L0_S12_F350',
        'L0_S10_F244',
        "magic4-1",
        'L3_S30_F3704_0.05700000002980232',
        'L3_S30_F3704_-0.25099998712539673',
        'L3_S30_F3704_-0.04500000178813934',
        'L3_S30_F3704_-0.1899999976158142',
        'L3_S30_F3704_0.01600000075995922',
        'L3_S30_F3704_-0.125',
        'L3_S30_F3704_-0.003000000026077032',
        'L3_S30_F3704_-0.004000000189989805',
        'L3_S30_F3704_-0.20999999344348907',
        'L3_S30_F3704_0.1979999989271164',
        'L3_S30_F3704_0.20100000500679016',
        'L3_S30_F3704_0.14000000059604645',
        'L3_S30_F3704_-0.08699999749660492',
        'L3_S30_F3704_0.05400000140070915',
        'L3_S30_F3704_0.017000000923871994',
        "L1_S24_F1844_-0.325_-0.275"
    ],
    [
        "hdf/train_numeric.hdf",
        "hdf/train_magic_numeric.hdf",
        "hdf/train_date_min_max.hdf",
        "hdf/train_date_L0_min_max.hdf",
        "hdf/train_date_L2_min_max.hdf",
        "hdf/train_date_L3_min_max.hdf",
        "hdf/train_magic.hdf",
        "hdf/train_categorical_L3_S32_F3854_decomposite.hdf",
        "hdf/train_S_min_max.hdf",
        "hdf/train_mask.hdf",
        "hdf/train_magic4-1.hdf",
        "hdf/train_L3_S30_F3704_deconposite.hdf",
        "hdf/train_L1_S24_F1844_-0.325_-0.275.hdf"
    ])

X = pd.concat((pd.read_hdf(filename)
               for filename in using_files[1]), axis=1).loc[:, using_files[0]]

columns, column_scores = select_ccolumns(X)
[(c, s)for c, s in zip(columns, column_scores)]
use_columns = [c for c, s in zip(columns, column_scores) if s > 0.005]

len(use_columns)

clf, best_threshold, predictions, column_names = xgboost_bosch(
    using_files[1],
    using_files[0])
g = xgboost.to_graphviz(clf.booster())
svg_str = g._repr_svg_()
mapper = {'f{0}'.format(i): v for i, v in enumerate(column_names)}
for k, v in mapper.items():
    svg_str = svg_str.replace(k + "&", v + "&")
with open("tree.svg", "w") as f:
    f.write(svg_str)
    f.close()

mapped = {mapper[k]: v for k, v in clf.booster().get_fscore().items()}
xgboost.plot_importance(mapped)


df = pd.read_hdf("hdf/train_response.hdf")
df["Predict"] = np.array(predictions > best_threshold).astype(np.int8)
df[(df["Response"] == 1) & (df["Predict"] == 0)].shape[0]
df[(df["Response"] == 0) & (df["Predict"] == 1)].shape[0]

test = pd.read_hdf("hdf/test_S29_C_md5_28.hdf").index
sub = predict(using_files[0], test, clf, best_threshold, using_files[1])
sub.to_csv("submission.csv.gz", compression="gzip")
