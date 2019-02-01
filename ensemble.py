import datetime
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from utils import make_submission

"""
ensembling using a convex combination of two models

TODO: 
- 
"""

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

# paths
ROOT = os.path.abspath('')
DATA_FOLDER = ROOT + '/data'
OUT_FOLDER = ROOT + '/out/ensemble'

# load data
df = pd.read_hdf(os.path.join(DATA_FOLDER, 'data_xgb_train.h5'), 'df')
val_1 = np.genfromtxt(os.path.join(DATA_FOLDER, 'for_ensemble/pred_val_xgb.csv'))
val_2 = np.genfromtxt(os.path.join(DATA_FOLDER, 'for_ensemble/pred_val_lgb.csv'))
test_1 = np.genfromtxt(os.path.join(DATA_FOLDER, 'for_ensemble/pred_test_xgb.csv'))
test_2 = np.genfromtxt(os.path.join(DATA_FOLDER, 'for_ensemble/pred_test_lgb.csv'))


X_train_level_2 = np.matrix([val_1, val_2]).T
Y_train_level_2 = np.array(df.loc[df['date_block_num'] == 33]['item_cnt_month'])
X_test_level_2 = np.matrix([test_1, test_2]).T

# check correlation between base learner predictions
np.corrcoef(X_train_level_2.T)
sns.jointplot(X_train_level_2[:,0], X_train_level_2[:,1])
plt.show()

# simple convex combination between pair
alphas_to_try = np.linspace(0, 1, 1001)
rmse_best = np.Inf
for alpha in alphas_to_try:
    mix = alpha * X_train_level_2[:,0] + (1-alpha) * X_train_level_2[:,1]
    rmse_new = np.sqrt(mean_squared_error(Y_train_level_2, mix))
    if rmse_new < rmse_best:
        alpha_best = alpha
        rmse_best = rmse_new

score = round(rmse_best, 6)
pred_test = alpha_best * X_test_level_2[:,0] + (1-alpha_best) * X_test_level_2[:,1]
ids = np.array(df.loc[df['date_block_num'] == 34, 'ID'])
submission = make_submission(ids, np.array(pred_test).flatten())

# export
today = datetime.datetime.now()
sub_id = today.strftime('%y%m%d') + '_' + today.strftime("%H%M") + \
		'_score_' + str(score)
folder = OUT_FOLDER + '/' + sub_id
os.mkdir(folder)
print('\n---- ' + sub_id + ' ----')
submission.to_csv(os.path.join(folder, 'submission.csv'), index=False)

