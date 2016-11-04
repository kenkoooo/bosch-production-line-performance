import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn import svm, model_selection
from sklearn.metrics import matthews_corrcoef, roc_auc_score, make_scorer
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import time
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


RESPONSE = "hdf/train_response.hdf"
files = [
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
    "hdf/train_L1_S24_F1844_-0.325_-0.275.hdf",
    "hdf/train_L3_S29_F3407_decomposite.hdf",
    "hdf/train_L3_S30_F3749_magic4-1.hdf",
    "hdf/train_min_max_30_50.hdf",
    "hdf/train_min_max_25.hdf",
    "hdf/train_mask_decomposite.hdf",
    "hdf/train_numeric_pattern_count.hdf",
    "hdf/train_F1723_-0.1.hdf",
    "hdf/train_F1844_6_magic4-1.hdf",
    "hdf/train_F1723_1844.hdf"]


def get_best_threshold(y, predictions):
    # MCC 最適化
    thresholds = np.linspace(0.01, 0.99, 200)
    mcc = np.array(
        [matthews_corrcoef(y, predictions > thr) for thr in thresholds])
    plt.plot(thresholds, mcc)
    best_threshold = thresholds[mcc.argmax()]
    print(mcc.max())
    return best_threshold

# 特徴抽出
X = pd.concat([pd.read_hdf(f) for f in files], axis=1)
X = X[X["magic4"] == -1]
y = pd.read_hdf(RESPONSE).loc[X.index].values.ravel()
column_names = X.columns

X = X.values

clf = XGBClassifier(max_depth=10, base_score=0.005, seed=71)
cv = StratifiedKFold(y, n_folds=3)
predictions = np.ones(y.shape[0])
for i, (train, test) in enumerate(cv):
    predictions[test] = clf.fit(
        X[train], y[train]).predict_proba(X[test])[:, 1]
    print("fold {}, ROC AUC: {:.3f}".format(
        i, roc_auc_score(y[test], predictions[test])))
print(roc_auc_score(y, predictions))
best_threshold = get_best_threshold(y, predictions)


tmp = [(clf.feature_importances_[i], column_names[i])
       for i in range(len(column_names))]
tmp = sorted(tmp, key=lambda x: x[0])
selected_columns = [a[1] for a in tmp if a[0] > 0.001]
[a for a in tmp if a[0] > 0.001]
