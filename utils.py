import numpy as np
import pandas as pd


"""

A bunch of functions used for feature generation:

 - mean_encoding(): mean encoding without regularization

"""

###################
# mean encoding
###################

def mean_encoding(df, cols, target, train_idx):
	train = df.loc[train_idx, cols + [target]]
	for col in cols:
		target_mean = train.groupby(col)[target].mean()
		df[col + '_enc'] = df[col].map(target_mean)
	return df