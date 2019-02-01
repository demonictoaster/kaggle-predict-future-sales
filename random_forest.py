import os
import pprint
import time

from hyperopt import fmin, tpe, hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns

from utils import *

"""
Random forest

NOTE:
- not used for final solution (ends up not being useful for ensembling since
  performance is not good enough and correlation of predictions with gradient
  boosted model is too high)

TODO: 
- 
"""

###################
# setup
###################

DEBUG = False  # if true take only subset of data to speed up computations
PARAM_OPT = True
PLOTS = False

n_trees = 100  # for 50 trees takes about 22min
param_opt_max_eval = 35

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

# paths
ROOT = os.path.abspath('')
DATA_FOLDER = ROOT + '/data'
OUT_FOLDER = ROOT + '/out/random_forest'

# import data
#df = pd.read_pickle(os.path.join(DATA_FOLDER, 'df.pkl'))
df = pd.read_hdf(os.path.join(DATA_FOLDER, 'data_xgb_train.h5'), 'df')

# shrink data size for debugging
if DEBUG==True:
	df = df.sample(frac=0.01, random_state=12)

###################
# prepare train / validation / test sets 
###################

# RF doesn't accept NaNs
df['price_diff_l1'].fillna(0, inplace=True)
df['price_diff_eom_l1'].fillna(0, inplace=True)
df['price_diff_eom_flag_l1'].fillna(-1, inplace=True)

# select columns to be used for training 
#print_columns_sorted(df)
cols_to_use = [
	# 'ID',
	# 'city_id',
	# 'city_id_month_avg_l1',
	# 'city_id_month_avg_l12',
	# 'city_id_month_avg_l2',
	# 'city_id_month_avg_l3',
	# 'city_id_month_avg_l6',
	'city_vs_cat_month_avg_l1',
	# 'date',
	# 'date_block_num',
	'item_category_id',
	'item_category_id_month_avg_l1',
	# 'item_category_id_month_avg_l12',
	# 'item_category_id_month_avg_l2',
	# 'item_category_id_month_avg_l3',
	# 'item_category_id_month_avg_l6',
	# 'item_cnt_month',
	'item_cnt_month_l1',
	# 'item_cnt_month_l12',
	'item_cnt_month_l2',
	'item_cnt_month_l3',
	'item_cnt_month_l6',
	'item_id',
	'item_id_month_avg_l1',
	# 'item_id_month_avg_l12',
	# 'item_id_month_avg_l2',
	# 'item_id_month_avg_l3',
	# 'item_id_month_avg_l6',
	'item_shop_sold_since',
	'item_sold_since',
	'item_subtype_id',
	'item_type_id',
	'item_vs_city_month_avg_l1',
	'month',
	# 'month_avg_l1',
	# 'n_days_in_month',
	# 'pair_not_in_train',
	# 'price_diff_l1',
	# 'price_diff_eom_l1',
	# 'price_diff_eom_flag_l1',
	# 'price_month_avg',
	# 'price_month_avg_diff_global_avg_l1',
	# 'price_month_avg_diff_last_six_month_l1',
	# 'price_month_avg_diff_prev_month_l1',
	# 'shop_id',
	'shop_id_month_avg_l1',
	# 'shop_id_month_avg_l12',
	# 'shop_id_month_avg_l2',
	# 'shop_id_month_avg_l3',
	# 'shop_id_month_avg_l6',
	'shop_vs_cat_month_avg_l1',
	'shop_vs_city_month_avg_l1',
	'shop_vs_item_month_avg_l1',
	# 'year'
]

# NOTE: train/val split is done consistently with train/test split 
# -> take last month of train data as validation set
X_train = df.loc[df['date_block_num'] < 33, cols_to_use]
Y_train = df.loc[df['date_block_num'] < 33]['item_cnt_month']
X_val = df.loc[df['date_block_num'] == 33, cols_to_use]
Y_val = df.loc[df['date_block_num'] == 33]['item_cnt_month']
X_test = df.loc[df['date_block_num'] == 34, cols_to_use]

###################
# hyper-param optimization 
###################

if PARAM_OPT == True:

	ts = time.time()

	def rf_loss(param):
		model = RandomForestRegressor(
			max_depth=param['max_depth'],
		    max_features=param['max_features'], 
		    min_samples_leaf=param['min_samples_leaf'],
		    n_estimators=n_trees,
		    n_jobs=4,    
		    random_state=12)

		model.fit(X_train, Y_train)
		loss = get_rmse(model, X_val, Y_val)

		print('Fitted RF using params:')
		pprint.pprint(param)
		print('\n--> Score = {0}'.format(loss))
		print('-----------------------------')
		return loss

	def rf_hyperopt():
	    space = {
		    'max_depth':  		hp.choice('max_depth', np.arange(1, 14, dtype=int)),
		    'max_features': 	hp.uniform('max_features', 0.3, 1),
		    'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 500, dtype=int))}

	    best = fmin(rf_loss, space, algo=tpe.suggest, max_evals=param_opt_max_eval)
	    return best

	best_model = rf_hyperopt()

	best_model = pd.DataFrame(best_model, index=[0]).T
	best_model.to_csv(os.path.join(OUT_FOLDER, 'random_forest_best_params.csv'), header=False)

	print('Best model:')
	pprint.pprint(best_model)
	spent = str(np.round((time.time() - ts) / 60, 2))
	print('\n---- Execution time: ' + spent + " min ----")


###################
# train model
###################

ts = time.time()
model = RandomForestRegressor(max_depth=10, 
							  max_features=0.34,
							  min_samples_leaf=42,
							  random_state=12, 
							  n_estimators=n_trees,
							  criterion='mse',
							  n_jobs=4,
							  verbose=1)

model.fit(X_train, Y_train)
spent = str(np.round((time.time() - ts) / 60, 2))
print('\n---- Execution time: ' + spent + " min ----")
os.system('say "Training over"')

###################
# plot
###################

if PLOTS == True:
	# plot feature importance
	importances = model.feature_importances_
	labels = cols_to_use
	plot_feature_importance(importances, labels)

###################
# predictions and export
###################

score = round(get_rmse(model, X_val, Y_val),6)
features = cols_to_use
params = model.get_params()
pred_val = model.predict(X_val).clip(0,20)
pred_test = model.predict(X_test).clip(0,20)
ids = np.array(df.loc[df['date_block_num'] == 34, 'ID'])
submission = make_submission(ids, pred_test)

if DEBUG==False:
	export_model(OUT_FOLDER, score, features, params, pred_val, pred_test, submission)

