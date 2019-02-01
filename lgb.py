import os
import pprint
import sys
import time

from hyperopt import fmin, tpe, hp
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

"""
Gradient boosted decision tree (LightGBM implementation)

TODO: 
- 
"""

###################
# setup
###################

DEBUG = False  # if true take only subset of data to speed up computations
PARAM_OPT = True
PLOTS = False  # display figures

param_opt_max_eval = 30  # 30 evals ~  

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

# paths
ROOT = os.path.abspath('')
DATA_FOLDER = ROOT + '/data'
OUT_FOLDER = ROOT + '/out/lightgbm'

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
	'city_vs_cat_month_avg_l1',
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
	# 'item_id_month_avg_l12',
	'item_id_month_avg_l2',
	# 'item_id_month_avg_l3',
	# 'item_id_month_avg_l6',
	'item_shop_sold_since',
	'item_sold_since',
	'item_subtype_id',
	'item_type_id',
	'item_vs_city_month_avg_l1',
	'month',
	'month_avg_l1',
	'n_days_in_month',
	# 'pair_not_in_train',
	'price_diff_l1',
	# 'price_diff_eom_l1',
	# 'price_diff_eom_flag_l1',
	# 'price_month_avg',
	'price_month_avg_diff_global_avg_l1',
	'price_month_avg_diff_last_six_month_l1',
	'price_month_avg_diff_prev_month_l1',
	'shop_id',
	'shop_id_month_avg_l1',
	# 'shop_id_month_avg_l12',
	'shop_id_month_avg_l2',
	# 'shop_id_month_avg_l3',
	# 'shop_id_month_avg_l6',
	'shop_vs_cat_month_avg_l1',
	'shop_vs_city_month_avg_l1',
	'shop_vs_item_month_avg_l1',
	'year'
]

# NOTE: train/val split is done consistently with train/test split 
# -> take last month of train data as validation set
X_train = df.loc[df['date_block_num'] < 33, cols_to_use]
Y_train = df.loc[df['date_block_num'] < 33]['item_cnt_month']
X_val = df.loc[df['date_block_num'] == 33, cols_to_use]
Y_val = df.loc[df['date_block_num'] == 33]['item_cnt_month']
X_test = df.loc[df['date_block_num'] == 34, cols_to_use]

train_data = lgb.Dataset(X_train, label=Y_train)
val_data = lgb.Dataset(X_val, label=Y_val)

###################
# hyper-param optimization 
###################

if PARAM_OPT == True:

	ts = time.time()

	def lgb_loss(param):
		lgb_params = {
		  'feature_fraction': param['feature_fraction'],
		  'metric': 'rmse',
		  'nthread':4,
		  'min_data_in_leaf': param['min_data_in_leaf'],
		  'bagging_fraction': param['bagging_fraction'],
		  'learning_rate': param['learning_rate'],
		  'objective': 'mse',
		  'bagging_seed': 12,
		  'num_leaves': param['num_leaves'],
		  'bagging_freq':1,
		  'verbose':0}

  		sys.stdout = open(os.devnull, "w")
		model = lgb.train(lgb_params, train_data, 300, 
			valid_sets=val_data, early_stopping_rounds=10)
		sys.stdout = sys.__stdout__

		loss = model.best_score['valid_0']['rmse']

		print('Fitted LightGBM using params:')
		pprint.pprint(param)
		print('\n--> Score = {0}'.format(loss))
		print('-----------------------------')
		return loss

	def lgb_hyperopt():
	    space = {
		    'num_leaves':  		hp.choice('num_leaves', np.arange(2, 100, dtype=int)),
		    'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(1, 500, dtype=int)),
		    'feature_fraction': hp.uniform('feature_fraction', 0.3, 1),
		    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
		    'learning_rate': 	hp.uniform('learning_rate', 0.025, 0.5)}

	    best = fmin(lgb_loss, space, algo=tpe.suggest, max_evals=param_opt_max_eval)
	    return best

	best_model = lgb_hyperopt()

	best_model = pd.DataFrame(best_model, index=[0]).T
	best_model.to_csv(os.path.join(OUT_FOLDER, 'lgb_best_params.csv'), header=False)

	print('Best model:')
	print(best_model)
	spent = str(np.round((time.time() - ts) / 60, 2))
	print('\nExecution time: ' + spent + " min")
	os.system('say "optimization over"')

###################
# train lightgbm
###################

lgb_params = {
  'feature_fraction': 0.8612037929669005,
  'metric': 'rmse',
  'nthread':4,
  'min_data_in_leaf': 427,
  'bagging_fraction': 0.5228483232848598,
  'learning_rate': 0.2210348619600142,
  'objective': 'mse',
  'bagging_seed': 12,
  'num_leaves': 24,
  'bagging_freq':1,
  'verbosity':1}

ts = time.time()
model = lgb.train(lgb_params, train_data, 300, valid_sets=val_data, early_stopping_rounds=10)

# print execution time
spent = str(np.round((time.time() - ts) / 60, 2))
print('\n---- Execution time: ' + spent + " min ----")
os.system('say "Training over"')

###################
# predictions and export
###################

score = round(model.best_score['valid_0']['rmse'],6)
features = cols_to_use
params = model.params
pred_val = model.predict(X_val, num_iteration=model.best_iteration).clip(0,20)
pred_test = model.predict(X_test, num_iteration=model.best_iteration).clip(0,20)
ids = np.array(df.loc[df['date_block_num'] == 34, 'ID'])
submission = make_submission(ids, pred_test)

if DEBUG==False:
	export_model(OUT_FOLDER, score, features, params, pred_val, pred_test, submission)
