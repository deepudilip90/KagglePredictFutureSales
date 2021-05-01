from os.path import join
import pandas as pd
from src.preprocess import preprocess

BASE_PATH = '/Users/deepudilip/ML/Coursera/HowToWinDSCompeition/Project'
DATA_PATH = join(BASE_PATH, 'data/')

def main():
    """:todo add docu"""
    item_cat_df = pd.read_csv(join(DATA_PATH, 'item_categories.csv'))
    item_df = pd.read_csv(join(DATA_PATH, 'items.csv'))
    sales_train_df = pd.read_csv(join(DATA_PATH, 'sales_train.csv'))
    shops_df = pd.read_csv(join(DATA_PATH, 'shops.csv'))
    test_df = pd.read_csv(join(DATA_PATH, 'test.csv'))

    sales_train_df = sales_train_df.merge(item_df, on='item_id', how='left')
    sales_train_df = sales_train_df.merge(item_cat_df, on='item_category_id', how='left')
    sales_train_df = sales_train_df.merge(shops_df, on='shop_id', how='left')

