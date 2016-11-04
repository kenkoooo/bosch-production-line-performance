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

RESPONSE = "hdf/train_response.hdf"


def get_best_threshold(y, predictions):
    # MCC 最適化
    thresholds = np.linspace(0.01, 0.99, 200)
    mcc = np.array(
        [matthews_corrcoef(y, predictions > thr) for thr in thresholds])
    plt.plot(thresholds, mcc)
    best_threshold = thresholds[mcc.argmax()]
    print(mcc.max())
    return best_threshold


def evalmcc_min(preds, dtrain):
    labels = dtrain.get_label()
    return 'MCC', -matthews_corrcoef(labels, preds > best_threshold)


def xgboost_bosch(X):
    y = pd.read_hdf(RESPONSE)
    y = y.loc[X.index].values.ravel()
    column_names = X.columns
    X = X.values
    # 予測して Cross Validation
    print("Predicting...")
    clf = XGBClassifier(max_depth=10, base_score=0.005, seed=71)
    cv = StratifiedKFold(y, n_folds=3)
    predictions = np.ones(y.shape[0])
    for i, (train, test) in enumerate(cv):
        predictions[test] = clf.fit(X[train],
                                    y[train],
                                    eval_metric=evalmcc_min).predict_proba(X[test])[:, 1]
        print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(y[
            test], predictions[test])))
    print(roc_auc_score(y, predictions))
    best_threshold = get_best_threshold(y, predictions)
    return clf, best_threshold, predictions, column_names

using_files = (
    [
        'L3_S30_F3704_-0.25099998712539673',
        'L3_S30_F3704_0.01600000075995922',
        'L3_S30_F3704_0.20100000500679016',
        'L3_S30_F3704_0.14000000059604645',
        'L3_S29_F3407_0.05700000002980232',
        'L3_S33_F3873',
        'L3_S30_F3704_-0.004000000189989805',
        'L3_S30_F3704_0.05400000140070915',
        'L3_S29_F3407_-0.028999999165534973',
        'min_max_30_50',
        'L3_S30_F3704_-0.125',
        'L3_S32_F3854_48',
        'L3_S30_F3704_0.017000000923871994',
        'L3_S30_F3704_-0.20999999344348907',
        'L3_S30_F3704_0.1979999989271164',
        'L3_S29_F3407_0.03700000047683716',
        'L3_S32_F3854_4',
        'mask',
        'L3_S30_F3704_-0.1899999976158142',
        'L3_S30_F3704_-0.04500000178813934',
        'L3_S38_F3956_magic4',
        'L3_S30_F3704_0.05700000002980232',
        'magic4-1',
        'L1_S24_F1844_-0.325_-0.275',
        'L3_max',
        'L3_S29_F3461',
        'L3_S38_F3952_magic4',
        'L3_S32_F3850_magic4',
        'L3_S32_F3854_2',
        'L3_S33_F3873_magic3',
        'L3_S29_F3407',
        'L1_S24_F1723_magic3',
        'L3_S38_F3960_magic4',
        'S32_min',
        'L3_min',
        'L1_S24_F1842',
        'L1_S24_F1844',
        'L3_S29_F3461_magic4',
        'L1_S24_F1846',
        'L0_S10_F244',
        'L0_S11_F322',
        'L0_S9_F180',
        'L0_S4_F104',
        'L3_S30_F3749_magic4',
        'L2_S26_F3040',
        'L3_S33_F3857_magic4',
        'max',
        'L0_S12_F350',
        'S33_min',
        'L3_S30_F3704',
        'L2_min',
        'L0_min',
        'magic4',
        'L3_S29_F3339',
        'L3_S29_F3351',
        'min',
        'L3_S30_F3829_magic3',
        'L3_S33_F3865',
        'L3_S30_F3744_magic3',
        'L3_S30_F3754_magic3',
        'L3_S33_F3859',
        'L3_S30_F3749',
        'L3_S33_F3857',
        'min_max',
        'mask_1001',
        'mask_111',
        'mask_1011',
        'mask_101',
        'mask_1',
        'mask_1111',
        'mask_1101',
        'mask_11',
        'mask_0',
        'mask_110',
        'mask_100',
        'mask_1010',
        'mask_1100'
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
        "hdf/train_L1_S24_F1844_-0.325_-0.275.hdf",
        "hdf/train_L3_S29_F3407_decomposite.hdf",
        "hdf/train_L3_S30_F3749_magic4-1.hdf",
        "hdf/train_min_max_30_50.hdf",
        "hdf/train_min_max_25.hdf",
        "hdf/train_mask_decomposite.hdf"
    ])

X = pd.concat((pd.read_hdf(filename)
               for filename in using_files[1]), axis=1).loc[:, using_files[0]]


clf, best_threshold, predictions, column_names = xgboost_bosch(X)


tmp = [(clf.feature_importances_[i], column_names[i])
       for i in range(len(column_names))]
tmp = sorted(tmp, key=lambda x: x[0])
tmp
mapped = {mapper[k]: v for k, v in clf.booster().get_fscore().items()}
xgboost.plot_importance(mapped)


X = pd.concat((pd.read_hdf(filename.replace("train", "test"))
               for filename in using_files[1]), axis=1).loc[:, using_files[0]]


test_indices = X.index
X = X.values
# 0 or 1 に正規化
best_threshold
predictions = (clf.predict_proba(X)[:, 1] > best_threshold).astype(np.int8)
# 提出データを生成
sub = pd.DataFrame(index=test_indices.astype(np.int32), columns=[])
sub["Response"] = predictions

sub.to_csv("submission.csv.gz", compression="gzip")
