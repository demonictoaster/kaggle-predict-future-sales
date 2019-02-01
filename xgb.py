import os
import pprint
import time

from hyperopt import fmin, tpe, hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from xgboost import plot_importance

from utils import *

"""
Gradient boosted decision tree (XGBoost implementation)

TODO: 
- 
"""

###################
# setup
###################

DEBUG = False  # if true take only subset of data to speed up computations
PARAM_OPT = False
PLOTS = False  # display figures

param_opt_max_eval = 30  # 30 evals ~ 240min  

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

# paths
ROOT = os.path.abspath('')
DATA_FOLDER = ROOT + '/data'
OUT_FOLDER = ROOT + '/out/xgb'

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
	# 'city_id_month_avg_l1',
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
	# 'item_id_month_avg_l12',
	'item_id_month_avg_l2',
	# 'item_id_month_avg_l3',
	# 'item_id_month_avg_l6',
	'item_shop_sold_since',
	'item_sold_since',
	'item_subtype_id',
	# 'item_type_id',
	# 'item_vs_city_month_avg_l1',
	'month',
	'month_avg_l1',
	# 'n_days_in_month',
	# 'pair_not_in_train',
	'price_diff_l1',
	# 'price_diff_eom_l1',
	# 'price_diff_eom_flag_l1',
	# 'price_month_avg',
	'price_month_avg_diff_global_avg_l1',
	# 'price_month_avg_diff_last_six_month_l1',
	# 'price_month_avg_diff_prev_month_l1',
	'shop_id',
	'shop_id_month_avg_l1',
	# 'shop_id_month_avg_l12',
	'shop_id_month_avg_l2',
	# 'shop_id_month_avg_l3',
	# 'shop_id_month_avg_l6',
	'shop_vs_cat_month_avg_l1',
	# 'shop_vs_city_month_avg_l1',
	# 'shop_vs_item_month_avg_l1',
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

	def xgb_loss(param):
		model = XGBRegressor(
			max_depth=param['max_depth'],
		    min_child_weight=param['min_child_weight'], 
		    gamma=param['gamma'],
		    colsample_bytree=param['colsample_bytree'], 
		    subsample=param['subsample'], 
		    eta=param['eta'],
		    n_estimators=1000,
		    n_jobs=4,    
		    seed=12)

		model.fit(
		    X_train, 
		    Y_train, 
		    eval_metric="rmse", 
		    eval_set=[(X_train, Y_train), (X_val, Y_val)], 
		    verbose=False, 
		    early_stopping_rounds = 10)

		loss = model.best_score

		print('Fitted XGB using params:')
		pprint.pprint(param)
		print('\n--> Score = {0}'.format(loss))
		print('-----------------------------')
		return loss

	def xgb_hyperopt():
	    space = {
		    'max_depth':  		hp.choice('max_depth', np.arange(3, 14, dtype=int)),
		    'min_child_weight': hp.quniform('min_child_weight', 1, 500, 1),
		    'gamma': 			hp.uniform('gamma', 0.5, 1),
		    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
		    'subsample': 		hp.uniform('subsample', 0.5, 1),
		    'eta': 				hp.uniform('eta', 0.025, 0.5)}

	    best = fmin(xgb_loss, space, algo=tpe.suggest, max_evals=param_opt_max_eval)
	    return best

	best_model = xgb_hyperopt()

	best_model = pd.DataFrame(best_model, index=[0]).T
	best_model.to_csv(os.path.join(OUT_FOLDER, 'xgb_best_params.csv'), header=False)

	print('Best model:')
	print(best_model)
	spent = str(np.round((time.time() - ts) / 60, 2))
	print('\nExecution time: ' + spent + " min")

###################
# final model 
###################

# define model
model = XGBRegressor(
	max_depth=7,
    min_child_weight=253, 
    colsample_bytree=0.6,
    gamma= 0.8,
    subsample=0.90, 
    eta=0.45,
    n_estimators=1000,
    n_jobs=4,    
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
	importance = model.feature_importances_
	plot_feature_importance(importance, cols_to_use)

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
# predictions and export
###################

score = model.best_score
features = cols_to_use
params = model.get_params()
pred_val = model.predict(X_val).clip(0,20)
pred_test = model.predict(X_test).clip(0,20)
ids = np.array(df.loc[df['date_block_num'] == 34, 'ID'])
submission = make_submission(ids, pred_test)

if DEBUG==False:
	export_model(OUT_FOLDER, score, features, params, pred_val, pred_test, submission)
