import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# params
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

# paths
ROOT = os.path.abspath('')
DATA_FOLDER = ROOT + '/data'

# load data
train = pd.read_pickle(os.path.join(DATA_FOLDER, 'train.pkl'))
test = pd.read_pickle(os.path.join(DATA_FOLDER, 'test.pkl'))

###################
# EDA on daily data
###################

train.item_cnt_day.describe()  # negative sales
train.item_price.describe()  # negative prices

# two item_cnt outliers
sns.set(color_codes=True)
sns.boxplot(train.item_cnt_day)
plt.show()

# one item_price outlier
sns.set(color_codes=True)
sns.boxplot(train.item_price)
plt.show()

# aggregate to monthly data
keys = ['date_block_num', 'shop_id', 'item_id']
df = train.groupby(keys, as_index=False).agg({
	'item_cnt_day':'sum',
	'item_price':'mean'})
df.rename(columns={'item_cnt_day':'item_cnt_month'}, inplace=True)

###################
# EDA on monthly data
###################

# all shops in test are in train but not all items 
# actually, test contains all item/shop pairs
all(elem in df.shop_id.unique() for elem in test.shop_id.unique())
all(elem in df.item_id.unique() for elem in test.item_id.unique())

# value_counts
df.shop_id.value_counts() 
df.item_id.value_counts()
test.item_id.value_counts()  # same number of observations for each item in test set!

# check outliers in item_cnt (-> some outliers!)
sns.boxplot(df.item_cnt_month)
plt.show()

# check outliers in price (-> one outlier)!
sns.boxplot(df.item_price)
plt.show()
df.item_price[np.argsort(df.item_price)[-10:]]
df.iloc[194725]  # check obesrvation
df.loc[df['item_id']==6066]  # check if only one item

# distribution by month
# g = sns.FacetGrid(df, col="date_block_num",  col_wrap=6)
# g = g.map(plt.hist, "item_cnt_month")
# plt.show()

# total sales by month -> we see peaks in December + downward trend
# November is usually second-highest selling month after December
tot_sales = df.groupby('date_block_num', as_index=False).agg({'item_cnt_month':'sum'})
sns.lineplot(x="date_block_num", y="item_cnt_month", data=tot_sales)
plt.show()

# scatter price vs sales (spot outlier in price)
sns.scatterplot(x="item_price", y="item_cnt_month", data=df)
plt.show()