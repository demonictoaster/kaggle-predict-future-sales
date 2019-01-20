import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance

import os
import time

"""
Fit XGBoost model

TODO: 
- tune hyperparameters
- try different early stopping methods
- save best model for ensembling 
"""

###################
# setup
###################

DEBUG = False  # if true take only subset of data to speed up computations

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

# paths
ROOT = os.path.abspath('')
DATA_FOLDER = ROOT + '/data'

# import data
df = pd.read_pickle(os.path.join(DATA_FOLDER, 'df.pkl'))

# shrink data size for debugging
if DEBUG==True:
	df = df.sample(frac=0.01, random_state=12)

###################
# prepare train / validation / test sets 
###################

# drop columns that cannot be used for training 
# (cannot use values from the future)
cols_to_drop = [
	'item_cnt_month', 
	'date', 
	'item_price', 
	'revenues', 
	'ID', 
	'city', 
	'item_price_diff',
	'item_on_sale',
	'item_price_diff_sign']

# show features we will use for training
df.drop(cols_to_drop, axis=1).info()

# NOTE: train/val split is done consistently with train/test split 
# -> take last month of train data as validation set
X_train = df.loc[df['date_block_num'] < 33].drop(cols_to_drop, axis=1)
Y_train = df.loc[df['date_block_num'] < 33]['item_cnt_month']
X_val = df.loc[df['date_block_num'] == 33].drop(cols_to_drop, axis=1)
Y_val = df.loc[df['date_block_num'] == 33]['item_cnt_month']
X_test = df.loc[df['date_block_num'] == 34].drop(cols_to_drop, axis=1)

###################
# XGBoost 
###################

model = XGBRegressor(
	max_depth=6,
    n_estimators=100,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

ts = time.time()
model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_val, Y_val)], 
    verbose=True, 
    early_stopping_rounds = 10)

# print execution time
spent = str(np.round((time.time() - ts) / 60, 2))
print('\n---- Execution time: ' + spent + " min ----")

# plot feature importance
plot_importance(model)
plt.show()

# predictions
pred = model.predict(X_test).clip(0,20)
ids = np.array(df.loc[df['date_block_num'] == 34, 'ID'])
submission = pd.DataFrame({
	'ID': ids.astype(np.int64),
	'item_cnt_month': pred
	})
submission.sort_values(by='ID', inplace=True)

if DEBUG==False:
	submission.to_csv(os.path.join(DATA_FOLDER, 'submission.csv'), index=False)



