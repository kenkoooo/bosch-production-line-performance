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
        "L1_S24_F1844_-0.325_-0.275",
        'L3_S29_F3407_0.05700000002980232',
        'L3_S29_F3407_0.03700000047683716',
        'L3_S29_F3407_-0.028999999165534973',
        "L3_S30_F3749_magic4-1",
        "min_max_30_50",
        "min_max_25"
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
        "hdf/train_min_max_25.hdf"
    ])

X = pd.concat((pd.read_hdf(filename)
               for filename in using_files[1]), axis=1).loc[:, using_files[0]]

X.where(np.isnan(X), 1.0, inplace=True)
X.where(X == 1.0, 0.0, inplace=True)
X.sum().sort_values()
