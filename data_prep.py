import gc
import itertools
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from utils import *

"""
NOTE:
- test set consists simply of all unique shop_id/item_id combinations (for that 
  specific month). Hence should format our train data accordingly. 

TODO:
- create feature new product (i.e. sold zero across all shops and then started selling)
  use lag for training/prediction
- add flag pair_present_in_train (if not probably means the shop doesn't even
  sell that product). If not helping much can hardcode predictiosn to be zero in these
  cases.
- check negative item prices (fill with median by block_id, shop_id and item_id)
- revenues by shop/date_block_num rather that raw
- make functions for interaction encoding
- try using only pairs present in test data rather than all pair for each month
- instead of casting to 16bit in the end, do it in the beginning to speed up things
- do some normalization (if plan to use linear models or knn)
- take logs and square transforms (for linear models)
- could experiment with groupings (e.g. items that sell a lot, or the ones that sell not much)
- create trend features (e.g. via moving averages)
- can bin some features and treate them as categorical
- use daily data to generate on_sale flag 
  (e.g. price change in last 10 days more than x %)
- price distance with respect to ther shops in last month
- create downcast function
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
# data import and monthly aggregation
###################

train = pd.read_pickle(os.path.join(DATA_FOLDER, 'train.pkl'))
test = pd.read_pickle(os.path.join(DATA_FOLDER, 'test.pkl'))

# aggregate to monthly data
keys = ['date_block_num', 'shop_id', 'item_id']
df = train.groupby(keys, as_index=False).agg({
	'item_cnt_day':'sum',
	'item_price':'mean'})
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
df['item_cnt_month'] = df['item_cnt_month'].fillna(0)

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

# winsorize price data to get rid of outlier in EDA and downcast to 16bit
df['item_price'] = stats.mstats.winsorize(df.item_price, limits=(0,0.01))

# append test set to train set to make lag creation easier
test['date_block_num'] = 34
test['date'] = "31.10.2015"
df = pd.concat([df, test], ignore_index=True, sort=False, 
		keys=keys)
df['ID'].fillna(-999, inplace=True)  # to be able to convert to integer
df['ID'] = df['ID'].astype(np.int32)

# downcast to 16 / 32 bits where possible
df['date_block_num'] = df['date_block_num'].astype(np.int16)
df['shop_id'] = df['shop_id'].astype(np.int16)
df['item_id'] = df['item_id'].astype(np.int16)
df['item_cnt_month'] = df['item_cnt_month'].astype(np.float32)
df['item_price'] = df['item_price'].astype(np.float32)
df['pair_not_in_train'] = df['pair_not_in_train'].astype(np.int16)

# lag item_cnt_month and item_price
df = make_lags(df, ['item_cnt_month'], lags)

# delete stuff we don't need anymore
del all_cbns, tmp, test
gc.collect()

###################
# feature generation
###################

# NOTE: scaling irrelevant for tree-based models

# revenues (in 000s)
# NOTE: price is average price during the month
df['revenues'] = df['item_price'] * df['item_cnt_month'] / 1000
df = make_lags(df, ['revenues'], [1])

# item category id
items = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
df = pd.merge(df, items.drop('item_name', axis=1), on='item_id')
df['item_category_id'] = df['item_category_id'].astype(np.int16)

# city can be retrieved from shop name (first string)
shops = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))	
shops['city'] = shops['shop_name'].str.split().str.get(0)
df = pd.merge(df, shops.drop('shop_name', axis=1), on='shop_id', how='left')
df['city_encoded'], _ = pd.factorize(df['city'])  # label encoding
df['city_encoded'] = df['city_encoded'].astype(np.int16)
df = df.drop('city', axis=1)

# month, year
df['year'] = pd.to_numeric(df['date'].astype(str).str[-4:])
df['month'] = pd.to_numeric(df['date'].astype(str).str[3:5])
df['year'] = df['year'].astype(np.int16)
df['month'] = df['month'].astype(np.int16)

# number of days in month (should have a small impact on sales)
df['n_days_in_month'] = 31.
df.loc[df['month'].isin([4, 6, 9, 11]), 'n_days_in_month'] = 30.
df.loc[df['month']==2, 'n_days_in_month'] = 28.  # no leap year
df['n_days_in_month'] = df['n_days_in_month'].astype(np.float32)

# delete stuff we don't need anymore
del items, shops
gc.collect()

###################
# price features
###################

tmp = df[['date_block_num', 'item_id', 'item_price']]

# global average price by item 
global_average = tmp.groupby('item_id', as_index=False).agg({'item_price': 'mean'})
global_average = global_average.rename(columns={'item_price': 'item_price_global_avg'})
tmp = pd.merge(tmp, global_average, on='item_id', how='left')

# monthly average price by item
month_average = tmp.groupby(['item_id', 'date_block_num'], as_index=False).agg({'item_price': 'mean'})
month_average = month_average.rename(columns={'item_price': 'item_price_month_avg'})
tmp = pd.merge(tmp, month_average, on=['date_block_num', 'item_id'], how='left')

# TODO: once done, dont' forget to deal with NaNs for price deltas

# delete stuff we don't need anymore
del tmp, global_average, month_average
gc.collect()

###################
# mean encoding
###################

# NOTE: should be calculated on train data only (exclude validation set)
# NOTE:  tradtional mean encoding doesn't work well
cols_to_mean_encode = [
	'shop_id',
	'item_id',
	'item_category_id',
	'city_encoded',
	'year'] # columns to mean-encode
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

###################
# interaction encoding
###################

# shop_id vs item_category
target_mean = df.groupby(['date_block_num', 'shop_id', 'item_category_id'], as_index=False)['item_cnt_month'].mean()
target_mean.rename(columns={'item_cnt_month': 'shop_vs_cat_month_avg'}, inplace=True)
df = pd.merge(df, target_mean, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')

# shop_id vs item_id
target_mean = df.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_cnt_month'].mean()
target_mean.rename(columns={'item_cnt_month': 'shop_vs_item_month_avg'}, inplace=True)
df = pd.merge(df, target_mean, on=['date_block_num', 'shop_id', 'item_id'], how='left')

# shop_id vs city
target_mean = df.groupby(['date_block_num', 'shop_id', 'city'], as_index=False)['item_cnt_month'].mean()
target_mean.rename(columns={'item_cnt_month': 'shop_vs_city_month_avg'}, inplace=True)
df = pd.merge(df, target_mean, on=['date_block_num', 'shop_id', 'city'], how='left')

# city vs item_category
target_mean = df.groupby(['date_block_num', 'city', 'item_category_id'], as_index=False)['item_cnt_month'].mean()
target_mean.rename(columns={'item_cnt_month': 'city_vs_cat_month_avg'}, inplace=True)
df = pd.merge(df, target_mean, on=['date_block_num', 'city', 'item_category_id'], how='left')

# month vs item_category
target_mean = df.groupby(['date_block_num', 'year', 'item_category_id'], as_index=False)['item_cnt_month'].mean()
target_mean.rename(columns={'item_cnt_month': 'year_vs_cat_month_avg'}, inplace=True)
df = pd.merge(df, target_mean, on=['date_block_num', 'year', 'item_category_id'], how='left')

to_lag = [
	'shop_vs_cat_month_avg',
	'shop_vs_item_month_avg',
	'shop_vs_city_month_avg',
	'city_vs_cat_month_avg',
	'year_vs_cat_month_avg']
df = make_lags(df, to_lag, [1])

# delete stuff we don't need anymore
del target_mean
gc.collect()

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
os.system('say "Data prep over"')
