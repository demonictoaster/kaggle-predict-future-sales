import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from xgboost import plot_importance

from utils import print_columns_sorted, plot_xgb_feature_importance

"""
Gradient boosted decision tree

TODO: 
- export some kind of log along with predictions to save param values
- hyperparameter tuning
- try different early stopping methods
- save best model for ensembling 
- for feature selection, can feed everything in a random forest and
  choose by feature importance
- make parameter files
- item_type_id weird importance
"""

###################
# setup
###################

DEBUG = False  # if true take only subset of data to speed up computations
PLOTS = False  # display figures

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

# paths
ROOT = os.path.abspath('')
DATA_FOLDER = ROOT + '/data'

# import data
#df = pd.read_pickle(os.path.join(DATA_FOLDER, 'df.pkl'))
df = pd.read_hdf(os.path.join(DATA_FOLDER, 'data_xgb_train.h5'), 'df')

# shrink data size for debugging
if DEBUG==True:
	df = df.sample(frac=0.01, random_state=12)

###################
# prepare train / validation / test sets 
###################

# select columns to be used for training 
# NOTE: cannot use information from the future
# NOTE: reducing the number of feature seems to lead to better generalization
#print_columns_sorted(df)
cols_to_use = [
	# 'ID',
	'city_id',
	'city_id_month_avg_l1',
	# 'city_id_month_avg_l12',
	# 'city_id_month_avg_l2',
	# 'city_id_month_avg_l3',
	# 'city_id_month_avg_l6',
	# 'city_vs_cat_month_avg_l1',
	# 'date',
	'date_block_num',
	'item_category_id',
	'item_category_id_month_avg_l1',
	# 'item_category_id_month_avg_l12',
	# 'item_category_id_month_avg_l2',
	# 'item_category_id_month_avg_l3',
	# 'item_category_id_month_avg_l6',
	# 'item_cnt_month',
	'item_cnt_month_l1',
	'item_cnt_month_l12',
	'item_cnt_month_l2',
	'item_cnt_month_l3',
	'item_cnt_month_l6',
	'item_id',
	'item_id_month_avg_l1',
	'item_id_month_avg_l12',
	'item_id_month_avg_l2',
	'item_id_month_avg_l3',
	'item_id_month_avg_l6',
	'item_shop_sold_since',
	'item_sold_since',
	'item_subtype_id',
	'item_type_id',
	'item_vs_city_month_avg_l1',
	'month',
	'month_avg_l1',
	'n_days_in_month',
	# 'pair_not_in_train',
	# 'price',
	# 'price_month_avg',
	'price_month_avg_diff_global_avg_l1',
	'price_month_avg_diff_last_six_month_l1',
	'price_month_avg_diff_prev_month_l1',
	'price_vs_month_avg_l1',
	# 'revenues_l1',
	'shop_id',
	'shop_id_month_avg_l1',
	'shop_id_month_avg_l12',
	'shop_id_month_avg_l2',
	'shop_id_month_avg_l3',
	'shop_id_month_avg_l6',
	'shop_vs_cat_month_avg_l1',
	# 'shop_vs_city_month_avg_l1',
	# 'shop_vs_item_month_avg_l1',
	# 'year',
	# 'year_vs_cat_month_avg_l1'
	]

# show features we will use for training
df[cols_to_use].info()

# NOTE: train/val split is done consistently with train/test split 
# -> take last month of train data as validation set
X_train = df.loc[df['date_block_num'] < 33, cols_to_use]
Y_train = df.loc[df['date_block_num'] < 33]['item_cnt_month']
X_val = df.loc[df['date_block_num'] == 33, cols_to_use]
Y_val = df.loc[df['date_block_num'] == 33]['item_cnt_month']
X_test = df.loc[df['date_block_num'] == 34, cols_to_use]

###################
# training 
###################

# define model
model = XGBRegressor(
	max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=12)

# train
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
os.system('say "Training over"')

###################
# some plots 
###################

if PLOTS==True:
	# plot feature importance
	plot_xgb_feature_importance(model, cols_to_use)

	# plot loss curves
	loss = model.evals_result()
	epochs = len(loss['validation_0']['rmse'])
	x_axis = range(0, epochs)
	plt.plot(x_axis, loss['validation_0']['rmse'], label='Train')
	plt.plot(x_axis, loss['validation_1']['rmse'], label='Test')
	plt.legend()
	plt.ylabel('RMSE')
	plt.show()

###################
# predictions 
###################
pred = model.predict(X_test).clip(0,20)
ids = np.array(df.loc[df['date_block_num'] == 34, 'ID'])
submission = pd.DataFrame({
	'ID': ids.astype(np.int64),
	'item_cnt_month': pred
	})
submission.sort_values(by='ID', inplace=True)

if DEBUG==False:
	submission.to_csv(os.path.join(DATA_FOLDER, 'submission.csv'), index=False)
