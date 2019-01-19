import pandas as pd

"""
Read the csv files and saves them as pickle files to speed up loading 
in later steps.
"""

# paths
ROOT = os.path.abspath('')
DATA_FOLDER = ROOT + '/data'

# for now don't use shop and category names
train = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
test = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv.gz'))

# save resulting dataframes
train.to_pickle(os.path.join(DATA_FOLDER, 'train.pkl'))
test.to_pickle(os.path.join(DATA_FOLDER, 'test.pkl'))