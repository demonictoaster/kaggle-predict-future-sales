import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

"""

A bunch of functions used for feature generation:

 - mean_encoding(): mean encoding without regularization
 - mean_encoding_kold(): mean encoding over k-folds
 - make_lags(): create lagged values for specified cols

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

def make_lags(df, cols, lags):
	for lag in lags:
		tmp = df[['shop_id', 'item_id', 'date_block_num'] + cols].copy()
		tmp['date_block_num'] += lag
		new_col_names = [col + "_l" + str(lag) for col in cols]
		tmp = tmp.rename(columns=dict(zip(cols, new_col_names)))
		df = pd.merge(df, tmp, on=['shop_id', 'item_id', 'date_block_num'], how='left')
	return df
