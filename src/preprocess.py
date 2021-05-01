import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess_shops(shops_df):
    """
    :todo add docu
    :param shops_df:
    :return:
    """
    # :todo does this need to be done on test data??
    shops_df.loc[shops_df['shop_name'] == 57, 'shop_name'] = 0
    shops_df.loc[shops_df['shop_name'] == 58, 'shop_name'] = 1
    shops_df.loc[shops_df['shop_name'] == 10, 'shop_name'] = 11

    shops_df['city'] = shops_df['shop_name'].apply(lambda x: x.split()[0])
    shops_df.loc[shops_df['city'] == '!Якутск', 'city'] = 'Якутск'
    label_encoder = LabelEncoder()
    shops_df['city_id'] = label_encoder.fit_transform(shops_df['city'])

    return shops_df


def preprocess_categories_df(item_cat_df):
    """
    :todo add docu
    :param item_cat_df:
    :return:
    """
    item_cat_df['split'] = item_cat_df['item_category_name'].str.split('-')
    item_cat_df['item_type'] = item_cat_df['split'].apply(lambda x: x[0].strip())
    item_cat_df['item_sub_type'] = item_cat_df['split'].apply(lambda x: x[1].strip() if len(x) > 1 else x[0])

    item_cat_df['item_type_id'] = LabelEncoder().fit_transform(item_cat_df['item_type'])
    item_cat_df['item_sub_type_id'] = LabelEncoder.fit_transform(item_cat_df['item_sub_type'])
    return item_cat_df


def preprocess(df, train_data=False, test_data=False):
    """
    :todo add docu
    :param df:
    :param train_data:
    :param test_data:
    :return:
    """
    if train_data:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day

        df = df[(df['item_price'] > 0) & (df['item_price'] < df['item_price'].quantile([.999]).tolist()[0])]
        df = df[(df['item_cnt_day'] > 0) & (df['item_cnt_day'] < df['item_cnt_day'].quantile([.999]).tolist()[0])]

    if test_data:
        pass






