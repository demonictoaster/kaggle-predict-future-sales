import numpy as np
import pandas as pd
import os

"""
Basic benchmark model that simply takes the average sales of the previous month
(by shop and item id) as the prediction for the new month. 
"""

# paths
ROOT = os.path.abspath('')
DATA_FOLDER = ROOT + '/data'

# load data
train = pd.read_pickle(os.path.join(DATA_FOLDER, 'train.pkl'))
test = pd.read_pickle(os.path.join(DATA_FOLDER, 'test.pkl'))

# aggregate to monthly data
keys = ['date_block_num', 'shop_id', 'item_id']
df = train.groupby(keys, as_index=False).agg({
	'item_cnt_day':'sum',
	'item_price':'mean'})
df.rename(columns={'item_cnt_day':'item_cnt_month'}, inplace=True)


###################
# benchmark model
###################

# try simple model taking sales from last month (October 2015)
oct15 = df.loc[(df['date_block_num']==33), :]

# merge with test set 
result = pd.merge(test, oct15, on=['shop_id', 'item_id'], how='left')

# fill NaN with zero
result['item_cnt_month'] = result['item_cnt_month'].fillna(0)

# clip values to [0,20] range
result['item_cnt_month'] = np.clip(result['item_cnt_month'], 0, 20)

# export submission (get 1.16777 MSE)
result[['ID', 'item_cnt_month']].to_csv(os.path.join(DATA_FOLDER, 'submission_bench.csv'), index=False)


