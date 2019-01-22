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
df = pd.read_pickle(os.path.join(DATA_FOLDER, 'df.pkl'))

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
	'item_category_id',
	'city_encoded',
	'year',
	'month',
	'n_days_in_month',
	# 'date_block_num_enc',
	# 'shop_id_enc',
	# 'item_id_enc',
	# 'item_category_id_enc',
	# 'city_encoded_enc',
	# 'year_enc',
	# 'month_enc',
	# 'date_block_num_enc_kfold',
	# 'shop_id_enc_kfold',
	# 'item_id_enc_kfold',
	# 'item_category_id_enc_kfold',
	# 'city_encoded_enc_kfold',
	# 'year_enc_kfold',
	# 'month_enc_kfold',
	'item_cnt_month_l1',
	'item_price_l1',
	'revenues_l1',
	'item_price_diff_l1',
	'item_on_sale_l1',
	'item_price_diff_sign_l1',
	'shop_id_month_sum_l1',
	'item_id_month_sum_l1',
	'item_category_id_month_sum_l1',
	'item_on_sale_month_sum_l1',
	# 'item_on_sale_enc_l1',
	# 'item_on_sale_enc_kfold_l1',
	'item_cnt_month_l2',
	'item_price_l2',
	'revenues_l2',
	'item_price_diff_l2',
	'item_on_sale_l2',
	'item_price_diff_sign_l2',
	'shop_id_month_sum_l2',
	'item_id_month_sum_l2',
	'item_category_id_month_sum_l2',
	'item_on_sale_month_sum_l2',
	# 'item_on_sale_enc_l2',
	# 'item_on_sale_enc_kfold_l2',
	'item_cnt_month_l3',
	'item_price_l3',
	'revenues_l3',
	'item_price_diff_l3',
	'item_on_sale_l3',
	'item_price_diff_sign_l3',
	'shop_id_month_sum_l3',
	'item_id_month_sum_l3',
	'item_category_id_month_sum_l3',
	'item_on_sale_month_sum_l3',
	# 'item_on_sale_enc_l3',	
	# 'item_on_sale_enc_kfold_l3'
	'item_cnt_month_l12',
	'item_price_l12',
	'revenues_l12',
	'item_price_diff_l12',
	'item_on_sale_l12',
	'item_price_diff_sign_l12',
	'shop_id_month_sum_l12',
	'item_id_month_sum_l12',
	'item_category_id_month_sum_l12',
	'item_on_sale_month_sum_l12',
	# 'item_on_sale_enc_l12'
	# 'item_on_sale_enc_kfold_l12'
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
	max_depth=7,
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
#os.system('say "Training over"')

###################
# some plots 
###################

if PLOTS==True:
	# plot feature importance
	plot_importance(model)
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


