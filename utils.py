import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.model_selection import KFold

"""

A bunch of functions used for feature generation:

 - mean_encoding(): mean encoding without regularization
 - mean_encoding_kold(): mean encoding over k-folds
 - mean_encoding_month(): mean envoding over month
 - make_lags(): create lagged values for specified cols by shop_id and item_id
 - make_lags_by): create lagged values for specified cols
 				  (time_idx and group_by to be specified manually)
 - print_columns_sorted(): prints column names conveniently

"""

###################
# mean encoding
###################

def mean_encoding(df, cols, target, train_idx):
	train = df.loc[train_idx, cols + [target]]
	global_avg = np.mean(train[target])
	for col in cols:
		target_mean = train.groupby(col)[target].mean()
		df[col + '_enc'] = df[col].map(target_mean)
		df[col + '_enc'].fillna(global_avg, inplace=True)
	return df

def mean_encoding_kfold(df, cols, target, train_idx, k):
	train = df.loc[train_idx, cols + [target]]
	global_avg = np.mean(train[target])
	kf = KFold(n_splits=k, shuffle=False)
	for col in cols:
		for idx_rest, idx_fold in kf.split(train):
			mean_fold = train.iloc[idx_rest].groupby(col)[target].mean()
			df.loc[df.index[idx_fold], col + '_enc_kfold'] = df[col].map(mean_fold)
			df[col + '_enc_kfold'].fillna(global_avg, inplace=True)
	return df

def mean_encoding_month(df, cols):
	for col in cols:
		target_mean = df.groupby(['date_block_num', col], as_index=False)['item_cnt_month'].mean()
		target_mean.rename(columns={'item_cnt_month': col + '_month_avg'}, inplace=True)
		df = pd.merge(df, target_mean, on=['date_block_num', col], how='left')
	return df

def make_lags(df, cols, lags):
	tmp = df[['shop_id', 'item_id', 'date_block_num'] + cols]
	for lag in lags:
		tmp_lagged = tmp.copy()
		tmp_lagged['date_block_num'] += lag
		new_col_names = [col + "_l" + str(lag) for col in cols]
		tmp_lagged = tmp_lagged.rename(columns=dict(zip(cols, new_col_names)))
		df = pd.merge(df, tmp_lagged, on=['shop_id', 'item_id', 'date_block_num'], how='left')
	return df

def make_lags_by(df, cols, lags, time_idx, by):
	tmp = df[time_idx + by + cols]
	for lag in lags:
		tmp_lagged = tmp.copy()
		tmp_lagged[time_idx] += lag
		new_col_names = [col + "_l" + str(lag) for col in cols]
		tmp_lagged = tmp_lagged.rename(columns=dict(zip(cols, new_col_names)))
		df = pd.merge(df, tmp_lagged, on=time_idx+by, how='left')
	return df

def print_columns_sorted(df):
	cols = sorted(df.columns.tolist())
	for col in cols:
		print('\'' + col + '\'' + ',')

def plot_xgb_feature_importance(booster,feature_names):
	features = feature_names
	importance = booster.feature_importances_
	to_plot = pd.DataFrame({'features': features, 'importance': importance})
	to_plot.sort_values('importance', ascending=False, inplace=True)
	sns.set(font_scale=0.8)
	plt.figure(figsize=(10,7))
	fig = sns.barplot(x='importance', y='features', data=to_plot)
	for item in fig.get_xticklabels():
		item.set_rotation(90)
	plt.subplot(111)
	plt.subplots_adjust(left=0.25, bottom=0.1, right=0.9, 
		 				top=0.9, wspace=0.05, hspace=0.05)
	plt.show()
