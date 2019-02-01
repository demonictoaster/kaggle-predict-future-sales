import datetime
import gc
import itertools
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer

from utils import *

"""
NOTE:
- test set consists simply of all unique shop_id/item_id combinations (for that 
  specific month). Hence should format our train data accordingly. 
- some of the data prep tricks were found in some kaggle kernels 
  (I would not have spotted the small Russian language subtleties!) 

TODO:
- could experiment with groupings (e.g. items that sell a lot, or the ones that sell not much)
- create trend features (e.g. via moving averages)
- can bin some features and treate them as categorical
- parameter file
"""

###################
# setup
###################

DEBUG = False  # if true take only subset of data to speed up computations
lags = [1, 2, 3, 6, 12]

ts = time.time()
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

# paths
ROOT = os.path.abspath('')
DATA_FOLDER = ROOT + '/data'

###################
# data import and cleaning
###################

train = pd.read_pickle(os.path.join(DATA_FOLDER, 'train.pkl'))
test = pd.read_pickle(os.path.join(DATA_FOLDER, 'test.pkl'))

# remove outliers in item_cnt and price
train = train[train['item_price'] < 100000]
train = train[train['item_cnt_day'] < 1001]

# for item with price<0, fill with median (of item on that month)
med = train.loc[(train['date_block_num']==4) & (train['shop_id']==32) & \
	(train['item_id']==2973), 'item_cnt_day'].median()
train.loc[train['item_price']<0, 'item_price'] = med

# fix duplicated shops
train.loc[train['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 0, 'shop_id'] = 57
train.loc[train['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 1, 'shop_id'] = 58
train.loc[train['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 10, 'shop_id'] = 11

###################
# monthly aggregation
###################

keys = ['date_block_num', 'shop_id', 'item_id']
df = train.groupby(keys, as_index=False).agg({'item_cnt_day':'sum'})
df.rename(columns={'item_cnt_day':'item_cnt_month'}, inplace=True)

# shrink data size for debugging
if DEBUG==True:
	df = df.sample(frac=0.01, random_state=12)
	test = test.sample(n=3, random_state=12)
	lags = [1]

###################
# data prep
###################

# we saw in first EDA step that test set contains all pairs for shop and item
# IDs that are present. Hence need to also reflect this in train set.
# we create all combinations for each month and merge that with main DF
all_cbns = []
for i in range(34):
	tmp = df[df['date_block_num']==i]
	labels = [tmp['date_block_num'].unique(),
		      tmp['shop_id'].unique(),
		      tmp['item_id'].unique()]
    cbns = (list(itertools.product(*labels)))
    all_cbns.append(np.array(cbns))

all_cbns = pd.DataFrame(np.vstack(all_cbns), columns=keys)
df = pd.merge(all_cbns, df, on=keys, how='left')

# set item_count to zero for non existing date/shop/item combinations
df['item_cnt_month'].fillna(0, inplace=True)

# expansion above yields some new shop/item pair that are not present in 
# original train data (across all months). 
# We can use this as a feature. Intuition is that shop probably doesn't
# sell the item if pair not present. 
# TODO: use full train set to for final submission
pairs_in_train = train.loc[train['date_block_num'] < 33, ['shop_id', 'item_id']].drop_duplicates(keep='first')
df = pd.merge(df, pairs_in_train, on=['shop_id', 'item_id'], how='left', indicator=True)
df.rename(columns={'_merge':'pair_not_in_train'}, inplace=True)
df['pair_not_in_train'], _ = pd.factorize(df['pair_not_in_train'])

# do the same for the test set
test = pd.merge(test, pairs_in_train, on=['shop_id', 'item_id'], how='left', indicator=True)
test.rename(columns={'_merge':'pair_not_in_train'}, inplace=True)
test['pair_not_in_train'], _ = pd.factorize(test['pair_not_in_train'])

# ground truth target values clipped into [0.20] range (see description on kaggle)
df.item_cnt_month = np.clip(df.item_cnt_month, 0, 20)

# create date for each date_block_num (set all to first day of the month)
tmp = []
for i in range(2013,2016):
	for j in range(1,13):
		tmp.append('01.%s.%s' %(str(j).zfill(2), i))
tmp = {'date_block_num':np.arange(34), 'date':tmp[0:34]}
tmp = pd.DataFrame(tmp)
df = pd.merge(df, tmp, on='date_block_num', how='left')

# append test set to train set to make lag creation easier
test['date_block_num'] = 34
test['date'] = "31.10.2015"
df = pd.concat([df, test], ignore_index=True, sort=False, 
		keys=keys)
df['ID'].fillna(-999, inplace=True)  # to be able to convert to integer
df['ID'] = df['ID'].astype(np.int32)

# downcast to 8- / 16- / 32- bit where possible
df = downcast(df)

# lag item_cnt_month and item_price
df = make_lags(df, ['item_cnt_month'], lags)

# delete stuff we don't need anymore
del all_cbns, tmp, test
gc.collect();

###################
# basic features
###################

# item category id
items = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
df = pd.merge(df, items.drop('item_name', axis=1), on='item_id')

# category type and subtype are included in item category name
# (found this on Kaggle)
cats = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
cats['split'] = cats['item_category_name'].str.split('-')
cats['item_type'] = cats['split'].map(lambda x: x[0].strip())
cats['item_type_id'], _ = pd.factorize(cats['item_type'])
cats['item_subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['item_subtype_id'], _ = pd.factorize(cats['item_subtype'])
cats = cats[['item_category_id', 'item_type_id', 'item_subtype_id']]
df = pd.merge(df, cats, on='item_category_id', how='left')

# city can be retrieved from shop name (first string)
# (found this on kaggle)
shops = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))	
shops['city'] = shops['shop_name'].str.split().str.get(0)
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

df = pd.merge(df, shops.drop('shop_name', axis=1), on='shop_id', how='left')
df['city_id'], _ = pd.factorize(df['city'])  # label encoding
df = df.drop('city', axis=1)

# month, year
df['year'] = pd.to_numeric(df['date'].astype(str).str[-4:])
df['month'] = pd.to_numeric(df['date'].astype(str).str[3:5])

# number of days in month (should have a small impact on sales)
df['n_days_in_month'] = 31.
df.loc[df['month'].isin([4, 6, 9, 11]), 'n_days_in_month'] = 30.
df.loc[df['month']==2, 'n_days_in_month'] = 28.  # no leap year

# first sale by item and by item/shop
df['item_first_sale'] = df.groupby('item_id')['date_block_num'].transform('min')
df['item_shop_first_sale'] = df.groupby(['item_id','shop_id'])['date_block_num'].transform('min')

# time since first sale
df['item_sold_since'] = df['date_block_num'] - df['item_first_sale']
df['item_shop_sold_since'] = df['date_block_num'] - df['item_shop_first_sale']
df = df.drop(['item_first_sale', 'item_shop_first_sale'], axis=1)

# downcast to 8- / 16- / 32- bit where possible
df = downcast(df)

# delete stuff we don't need anymore
del items, cats, shops
gc.collect();

###################
# on sale flag
###################

# price changes (take log returns so can sum returns for aggregation)
train['date_id'] = train['date'].str.split(pat='.')
train['date_id'] = train['date_id'].apply(lambda x:x[2]) + \
				   train['date_id'].apply(lambda x:x[1]) + \
				   train['date_id'].apply(lambda x:x[0])
train['date_id'] = train['date_id'].astype(np.int16)
train.sort_values(by=['shop_id', 'item_id', 'date_id'], inplace=True)
train['price_prev'] = train.groupby(['shop_id', 'item_id'])['item_price'].shift()
train['price_diff'] = np.log(train['item_price'] / train['price_prev'])
train['price_diff'].fillna(0, inplace=True)

# since we make forecasts for next month, only changes at end of month mater
# (assumption is that price cuts will matter only in the next few days)
train['day'] = pd.to_numeric(train['date'].astype(str).str[:2])
train['price_diff_eom'] = np.where(train['day'] > 24, train['price_diff'], 0)
train['price_diff_eom_flag'] = np.where(train['price_diff_eom'] < -0.2, 1, -1)

tmp = train.groupby(keys, as_index=False).agg({'price_diff':'sum', 
											   'price_diff_eom': 'sum',
											   'price_diff_eom_flag': 'max'})
df = pd.merge(df, tmp, on=keys, how='left')

df = make_lags(df, ['price_diff', 'price_diff_eom', 'price_diff_eom_flag'], [1])
df = df.drop(['price_diff', 'price_diff_eom', 'price_diff_eom_flag'], axis=1)

# delete stuff we don't need anymore
del tmp
gc.collect(); 

###################
# price features 
###################

# global average price by item 
global_avg = train.groupby('item_id', as_index=False).agg({'item_price': 'mean'})
global_avg = global_avg.rename(columns={'item_price': 'price_global_avg'})

# monthly average price by item and 6 month lags
month_avg = train.groupby(['item_id', 'date_block_num'], as_index=False).agg({'item_price': 'mean'})
month_avg = month_avg.rename(columns={'item_price': 'price_month_avg'})
month_avg = make_lags_by(df=month_avg, 
				   cols=['price_month_avg'], 
				   lags=[1,2,3,4,5,6], 
				   time_idx=['date_block_num'], 
				   by=['item_id'])

# difference in last 6 months by item (use available prices in that range)
# NOTE: a bit slow, maybe there is a faster way...
def price_diff_in_last_six_month(row):
	row = row.copy()
	row.sort_index(ascending=True, inplace=True)
	recent = row.loc[('price' in s for s in row.index)].first_valid_index()
	old = row.loc[('price' in s for s in row.index)].last_valid_index()
	if old is None:
		return 0
	if row[old] != 0:
		diff = row[recent] / row[old] - 1
		return diff
	else:
		return 0

month_avg['price_month_avg_diff_last_six_month'] = \
	month_avg.apply(price_diff_in_last_six_month, axis=1)

# difference with respect to previous month by item
month_avg['price_month_avg_diff_prev_month'] = \
	month_avg['price_month_avg'] / month_avg['price_month_avg_l1'] - 1

# difference between monthly average and global average
month_avg = pd.merge(month_avg, global_avg, on='item_id', how='left')
month_avg['price_month_avg_diff_global_avg'] = \
	month_avg['price_month_avg'] / month_avg['price_global_avg'] - 1

# keep relevant columns and merge with main dataframe
month_avg = month_avg[[
	'item_id', 
	'date_block_num',
	'price_month_avg', 
	'price_month_avg_diff_prev_month',
	'price_month_avg_diff_last_six_month',
	'price_month_avg_diff_global_avg']]
df = pd.merge(df, month_avg, on=['date_block_num', 'item_id'], how='left')

# downcast to 8- / 16- / 32- bit where possible
df = downcast(df)

# create lags and get rid of what we don't need
to_lag = [
	'price_month_avg_diff_prev_month',
	'price_month_avg_diff_last_six_month',
	'price_month_avg_diff_global_avg']
df = make_lags_by(df=df, cols=to_lag, lags=[1], 
	time_idx=['date_block_num'], by=['shop_id','item_id'])
df = df.drop(to_lag, axis=1)

# deal with NaNs for price deltas
cols = [col + '_l1' for col in to_lag]
for col in cols:
	df[col].fillna(0, inplace=True)

# delete stuff we don't need anymore
del global_avg, month_avg, to_lag
gc.collect();



###################
# mean encoding
###################

# by date
mean_by_date = df.groupby('date_block_num', as_index=False)['item_cnt_month'].mean()
mean_by_date.rename(columns={'item_cnt_month': 'month_avg'}, inplace=True)
df = pd.merge(df, mean_by_date, on='date_block_num', how='left')
df = make_lags(df, ['month_avg'], [1])

# by date and some categorical feature
cols_to_mean_encode = [
	'shop_id',
	'item_id',
	'item_category_id',
	'city_id'] # columns to mean-encode
target = 'item_cnt_month'
train_idx = df['date_block_num'].isin(np.arange(0, 33))
k = 5

# df = mean_encoding(df, cols_to_mean_encode, target, train_idx)  # no regularization
# df = mean_encoding_kfold(df, cols_to_mean_encode, target, train_idx, k=5)  # over k-folds
df = mean_encoding_month(df, cols_to_mean_encode)
df = make_lags(df, [col + '_month_avg' for col in cols_to_mean_encode], lags)

# check correlation with target variable
# for col in cols_to_mean_encode:
# 	cor = np.corrcoef(df.loc[train_idx, col + '_month_avg'].values, df.loc[train_idx, 'item_cnt_month'].values)[0][1]
# 	print(cor)

# drop columns we don't need
to_drop = ['month_avg'] + [col + '_month_avg' for col in cols_to_mean_encode]
df = df.drop(to_drop, axis=1)

# delete stuff
del mean_by_date, cols_to_mean_encode
gc.collect();

###################
# interaction encoding
###################

def mean_encoding_by(df, target, by, new_var_name):
	target_mean = df.groupby(by, as_index=False)[target].mean()
	target_mean.rename(columns={target: new_var_name}, inplace=True)
	df = pd.merge(df, target_mean, on=by, how='left')
	return df

df = mean_encoding_by(df, 'item_cnt_month', 
	['date_block_num', 'shop_id', 'item_category_id'], 'shop_vs_cat_month_avg')

df = mean_encoding_by(df, 'item_cnt_month', 
	['date_block_num', 'shop_id', 'item_id'], 'shop_vs_item_month_avg')

df = mean_encoding_by(df, 'item_cnt_month', 
	['date_block_num', 'shop_id', 'city_id'], 'shop_vs_city_month_avg')

df = mean_encoding_by(df, 'item_cnt_month', 
	['date_block_num', 'city_id', 'item_category_id'], 'city_vs_cat_month_avg')

df = mean_encoding_by(df, 'item_cnt_month', 
	['date_block_num', 'item_id', 'city_id'], 'item_vs_city_month_avg')

to_lag = [
	'shop_vs_cat_month_avg',
	'shop_vs_item_month_avg',
	'shop_vs_city_month_avg',
	'city_vs_cat_month_avg',
	'item_vs_city_month_avg']
df = make_lags(df, to_lag, [1])

# drop columns we don't need
to_drop = to_lag
df = df.drop(to_drop, axis=1)

# delete stuff we don't need anymore
del to_drop, to_lag
gc.collect();

###################
# last steps and save
###################

# lags lead to NaNs, need to replace by zero for all variables based on item_cnt_month
for col in df.columns:
	if ('item_cnt_month' in col or 'revenues' in col or 'month_avg' in col):
		df[col].fillna(0, inplace=True)

# remove periods for which lags cannot be computed (date_block_num starts at 0)
# NOTE: lose 12 months of data if maxlag=12, might not be worth to include
df = df[df['date_block_num'] >= max(lags)]

# write dataframe to folder
df.info()
if DEBUG==False:
	#df.to_pickle(os.path.join(DATA_FOLDER, 'df.pkl'))
	df.to_hdf(os.path.join(DATA_FOLDER, 'data_xgb_train.h5'), key='df', mode='w')


# execution time
spent = str(np.round((time.time() - ts) / 60, 2))
print('\n---- Execution time: ' + spent + " min ----")
os.system('say "Data prep over"');
