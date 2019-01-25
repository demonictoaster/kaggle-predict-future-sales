import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance

"""
Gradient boosted decision tree

TODO: 
- export some kind of log along with predictions to save param values
- hyperparameter tuning
- try different early stopping methods
- save best model for ensembling 
- for feature selection, can feed everythin in a random forest and
  choose by feature importance
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
cols_to_use = [
	'date_block_num',
	'shop_id',
	'item_id',
	# 'pair_not_in_train',
	'item_category_id',
	'city_encoded',
	'year',
	'month',
	'n_days_in_month',
	'item_cnt_month_l1',
	# 'item_price_l1',
	# 'revenues_l1',
	# 'item_price_diff_l1',
	# 'item_on_sale_l1',
	# 'item_price_diff_sign_l1',
	'shop_id_month_avg_l1',
	'item_id_month_avg_l1',
	'item_category_id_month_avg_l1',
	'city_encoded_month_avg_l1',
	'year_month_avg_l1',
	# 'item_on_sale_month_avg_l1',
	'shop_vs_cat_month_avg_l1',
	# 'shop_vs_item_month_avg_l1',
	'shop_vs_city_month_avg_l1',
	'city_vs_cat_month_avg_l1',
	'year_vs_cat_month_avg_l1',
	'item_cnt_month_l2',
	# 'item_price_l2',
	# 'revenues_l2',
	'shop_id_month_avg_l2',
	'item_id_month_avg_l2',
	# 'item_category_id_month_avg_l2',
	# 'city_encoded_month_avg_l2',
	# 'year_month_avg_l2',
	# 'item_on_sale_month_avg_l2',
	# 'shop_vs_cat_month_avg_l2',
	# 'shop_vs_item_month_avg_l2',
	# 'shop_vs_city_month_avg_l2',
	# 'city_vs_cat_month_avg_l2',
	# 'year_vs_cat_month_avg_l2',
	'item_cnt_month_l3',
	# 'item_price_l3',
	# 'revenues_l3',
	'shop_id_month_avg_l3',
	'item_id_month_avg_l3',
	# 'item_category_id_month_avg_l3',
	# 'city_encoded_month_avg_l3',
	# 'year_month_avg_l3',
	# 'item_on_sale_month_avg_l3',
	# 'shop_vs_cat_month_avg_l3',
	# 'shop_vs_item_month_avg_l3',
	# 'shop_vs_city_month_avg_l3',
	# 'city_vs_cat_month_avg_l3',
	# 'year_vs_cat_month_avg_l3',
	'item_cnt_month_l6',
	# 'item_price_l6',
	# 'revenues_l6',
	'shop_id_month_avg_l6',
	'item_id_month_avg_l6',
	# 'item_category_id_month_avg_l6',
	# 'city_encoded_month_avg_l6',
	# 'year_month_avg_l6',
	# 'item_on_sale_month_avg_l6',
	# 'shop_vs_cat_month_avg_l6',
	# 'shop_vs_item_month_avg_l6',
	# 'shop_vs_city_month_avg_l6',
	# 'city_vs_cat_month_avg_l6',
	# 'year_vs_cat_month_avg_l6',
	'item_cnt_month_l12',
	# 'item_price_l12',
	# 'revenues_l12',
	'shop_id_month_avg_l12',
	'item_id_month_avg_l12',
	# 'item_category_id_month_avg_l12',
	# 'city_encoded_month_avg_l12',
	# 'year_month_avg_l12',
	# 'item_on_sale_month_avg_l12',
	# 'shop_vs_cat_month_avg_l12',
	# 'shop_vs_item_month_avg_l12',
	# 'shop_vs_city_month_avg_l12',
	# 'city_vs_cat_month_avg_l12',
	# 'year_vs_cat_month_avg_l12'
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
	fig = plot_importance(model)
	plt.show()

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
