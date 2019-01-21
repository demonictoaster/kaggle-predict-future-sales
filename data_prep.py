import itertools
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from utils import *

"""
NOTE:
- no feature in test since forecasting -> need to use lags for all variables
- can also use historical averages lags
- test set is simply all unique shop_id/item_id combinations

TODO
- look at outliers (probably need to check by product and by shop,
  cannot blindly winsorize over whole sample)
- do some normalization (if plan to use linear models or knn)
- create lags
- take logs and square transforms (for linear models)
- we know that test data is for November 2015. Can use that as feature
  (probably more sales in November/december than other months 
  since Christmas period!)
- maybe worth discarding shops that have just very few sales??
- might be worth backfilling the train set with all dates
  and set zeros when there were no sales
- use label encoder (.factorize()) for dtype=='object'
- use .feature_importance on trained model to see which ones are important
- could experiment with groupings (e.g. items that sell a lot, or the ones that sell not much)
- could check changes in item prices to see if item is on sales
  could also use a feature "on_sale_in_prev_month" (sales should decrease after)
- should also be able to use item price as feature
  (assuming it is the same as previous month, or same as in November 2014)
- create trend features (e.g. via moving averages)
- average sales in past month, number of sales in same month previous year
- check distribution of shop_id and item_id across train and test and make
  sure valiation set has same distribution as test set
- check if test is simply all combinations of shop_id and test_id.
  In this case we don't have same distrib in train and test (and predictions
  should be zero for combinations that are not in train data). In this case
  we must complete the validation set with combinations that have zero sales. 
- could also use shop names (to flag shop that are part of a chain)
- add trend variable
- transform int and float to 16 bit to gain space
- try using only pairs present in test data rather than all pair for each month
"""

###################
# setup
###################

DEBUG = True  # if true take only subset of data to speed up computations

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
	test = test.sample(n=10, random_state=12)

###################
# data prep
###################

# we saw in first EDA step that test set contains all pairs for shop and item
# IDs that are present. Hence need to also reflect this in train set.

# first we create all combinations for each month and store them in a list
all_cbns = []
for i in range(34):
	tmp = df[df['date_block_num']==i]
	labels = [tmp['date_block_num'].unique(),
		      tmp['shop_id'].unique(),
		      tmp['item_id'].unique()]
    cbns = (list(itertools.product(*labels)))
    all_cbns.append(np.array(cbns))

# create pandas DF out of this list
all_cbns = pd.DataFrame(np.vstack(all_cbns), columns=keys)

# merge with train data
df = pd.merge(all_cbns, df, on=keys, how='left')

# for non existing pair in train, item_count should be set to zero
df['item_cnt_month'] = df['item_cnt_month'].fillna(0)

# create date for each date_block_num (set all to first day of the month)
tmp = []
for i in range(2013,2016):
	for j in range(1,13):
		tmp.append('01.%s.%s' %(str(j).zfill(2), i))
tmp = {'date_block_num':np.arange(34), 'date':tmp[0:34]}
tmp = pd.DataFrame(tmp)
df = pd.merge(df, tmp, on='date_block_num', how='left')

# ground truth target values clipped into [0.20] range (see description on kaggle)
df.item_cnt_month = np.clip(df.item_cnt_month, 0, 20)

# winsorize price data to get rid of outlier in EDA
df['item_price'] = stats.mstats.winsorize(df.item_price, limits=(0,0.01))

# append test set to train set to make lag creation easier
test['date_block_num'] = 34
test['date'] = "31.10.2015"
df = pd.concat([df, test], ignore_index=True, sort=False, 
		keys=keys)

# delete stuff we don't need anymore
del all_cbns, tmp, test


###################
# feature generation
###################

# NOTE: scaling irrelevant for tree-based models

# sales 
# NOTE: price is average price during the month
df['revenues'] = df['item_price'] * df['item_cnt_month']

# item category id
items = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
df = pd.merge(df, items.drop('item_name', axis=1), on='item_id')
del items

# shops info: read on forum that first word is the city
# (hard to guess if you don't speak Russian...)
shops = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))	
shops['city'] = shops['shop_name'].str.split().str.get(0)
df = pd.merge(df, shops.drop('shop_name', axis=1), on='shop_id', how='left')
df['city_enc'], _ = pd.factorize(df['city'])  # label encoding
del shops

# month, year
df['year'] = pd.to_numeric(df['date'].astype(str).str[-4:])
df['month'] = pd.to_numeric(df['date'].astype(str).str[3:5])

# number of days in month (should have a small impact on sales)
df['n_days_in_month'] = 31.
df.loc[df['month'].isin([4, 6, 9, 11]), 'n_days_in_month'] = 30.
df.loc[df['month']==2, 'n_days_in_month'] = 28.  # no leap year

# change in price relative to previous month + on_sale flag
df.sort_values(['shop_id','item_id', 'date_block_num'], inplace=True)
df['item_price_diff'] = df.groupby(['shop_id', 'item_id'])['item_price'].diff()
df['item_on_sale'], _ = pd.factorize(df['item_price_diff'] < 0)
df['item_price_diff_sign'] = np.sign(df['item_price_diff'].fillna(0)).astype(int)

# use daily data to generate on_sale flag (e.g. price change in last 10 days)



# mean over previous months (e.g. over 1, 2, 3 previous months)
# for sales, item_cnt, etc. 

# can bin some features and treate them as categorical

# lagged values of total shop or total item sales

###################
# mean encoding
###################

# NOTE: should be calculated on train data only (exclude validation set)
# TODO: recalculate on full dataset for final model

cols = [
	'date_block_num',
	'shop_id',
	'item_id',
	'item_category_id',
	'city_enc',
	'year',
	'month',
	'item_on_sale'] # columns to mean-encode
target = 'item_cnt_month'
train_idx = df['date_block_num'].isin(np.arange(0, 33))

df = mean_encoding(df, cols, target, train_idx)  # no regularization



###################
# create lags
###################

def make_lags(df, cols, lags):
	for lag in lags:
		tmp = df[['shop_id', 'item_id', 'date_block_num'] + cols].copy()
		tmp['date_block_num'] += lag
		new_col_names = [col + "_l" + str(lag) for col in cols]
		tmp = tmp.rename(columns=dict(zip(cols, new_col_names)))
		df = pd.merge(df, tmp, on=['shop_id', 'item_id', 'date_block_num'], how='left')
		print("lag %d created" %lag)
	return(df)

lags = [1, 3, 12]
cols_to_lag = [
	'item_cnt_month', 
	'item_price', 
	'revenues',
	'item_price_diff',
	'item_on_sale',
	'item_price_diff_sign']
df = make_lags(df, cols_to_lag, lags)

# remove periods for which lags cannot be computed (date_block_num starts at 0)
df = df[df['date_block_num'] >= max(lags)]

# replace null by zero for target
for col in df.columns:
	if ('item_cnt_month_l' in col):
		df[col].fillna(0, inplace=True)

# further variables based on lags (differences)

# save to pickle
if DEBUG==False:
	df.to_pickle(os.path.join(DATA_FOLDER, 'df.pkl'))

# print execution time
spent = str(np.round((time.time() - ts) / 60, 2))
print('\n---- Execution time: ' + spent + " min ----")


