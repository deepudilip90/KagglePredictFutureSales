import pandas as pd
import numpy as np
from itertools import product
from os import path

from sklearn.preprocessing import LabelEncoder

BASE_PATH = '/Users/deepudilip/ML/Coursera/HowToWinDSCompeition/KagglePredictFutureSales'
DATA_PATH = path.join(BASE_PATH, 'data/')

def preprocess_shops_df(shops_df):
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
    shops_df = shops_df[['shop_id', 'city_id']]

    shops_df['shop_id'] = shops_df['shop_id'].astype(np.int16)
    shops_df['city_id'] = shops_df['city_id'].astype(np.int16)

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
    item_cat_df = item_cat_df[['item_category_id', 'item_type_id', 'item_sub_type_id']]

    item_cat_df['item_category_id'] = item_cat_df['item_category_id'].astype(np.int16)
    item_cat_df['item_type_id'] = item_cat_df['item_type_id'].astype(np.int16)
    item_cat_df['item_sub_type_id'] = item_cat_df['item_sub_type_id'].astype(np.int16)

    return item_cat_df


def preprocess_item_df(item_df):
    """
    :todo add docu
    :param item_df:
    :return:
    """
    item_df = item_df.drop(['item_name'], axis=1)
    item_df['item_id'] = item_df['item_id'].astypy(np.int16)

    return item_df


def preprocess(df, only_train=False, test_data=False):
    """
    :todo add docu
    :param df:
    :param only_train:
    :param test_data:
    :return:
    """
    if only_train:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day

        df = df[(df['item_price'] > 0) & (df['item_price'] < df['item_price'].quantile([.999]).tolist()[0])]
        df = df[(df['item_cnt_day'] > 0) & (df['item_cnt_day'] < df['item_cnt_day'].quantile([.999]).tolist()[0])]

    if test_data:
        pass


def prepare_train_df(sales_train_df, train_months=34):
    """
    :todo add docu
    :param sales_train_df:
    :return:
    """
    data_matrix = []
    cols = ['date_block_num', 'shop_id', 'item_id']
    for i in range(train_months):
        sales = sales_train_df[sales_train_df['date_block_num'] == i]
        data_matrix.append(list(product([i], sales.shop_id.unique(), sales.item_id.unique())))

    data_matrix = pd.DataFrame(np.vstack(data_matrix), columns=cols)
    data_matrix['date_block_num'] = data_matrix['date_block_num'].astype(np.int8)
    data_matrix['shop_id'] = data_matrix['shop_id'].astype(np.int8)
    data_matrix['item_id'] = data_matrix['item_id'].astype(np.int16)
    data_matrix.sort_values(cols, inplace=True)

    sales_train_monthly = sales_train_df.groupby(['date_block_num',
                                                  'shop_id',
                                                  'item_id']).agg({'item_cnt_day': 'sum'})
    sales_train_monthly.columns = ['item_cnt_month']
    sales_train_monthly.reset_index(inplace=True)

    data_matrix = (data_matrix.merge(sales_train_monthly,
                                     on=['date_block_num',
                                         'shop_id',
                                         'item_id'],
                                     how='left')
                   .fillna(0))
    data_matrix['item_cnt_month'] = data_matrix['item_cnt_month'].clip(0, 20)

    return data_matrix

def set_data_types(df, col_types_dict):
    """
    Convenience function to set data types of a set of columns in the dataframe as per the definition provided in the
    input parameter col_types_dict.

    :param df: Dataframe with the columns for which data types need to be altered.
    :type df: pandas.DataFrame
    :param col_types_dict: A dictionary with keys as the column names and the values as the datatype. Allowed values for
                           datatypes are 'int16', 'int32', 'float'.
    :return: Dataframe with the data types of the column changed.
    """
    pass


def main():
    """
    Main method to run the preprocessing steps
    :return:
    """

    item_cat_df = pd.read_csv(path.join(DATA_PATH, 'item_categories.csv'))
    item_df = pd.read_csv(path.join(DATA_PATH, 'items.csv'))
    sales_train_df = pd.read_csv(path.join(DATA_PATH, 'sales_train.csv'))
    shops_df = pd.read_csv(path.join(DATA_PATH, 'shops.csv'))
    test_df = pd.read_csv(path.join(DATA_PATH, 'test.csv'))

    shops_df = preprocess_shops_df(shops_df)
    item_cat_df = preprocess_categories_df(item_cat_df)
    item_df = preprocess_item_df(item_df)

    train_df = prepare_train_df(sales_train_df, train_months=34)
    train_df = train_df.merge(shops_df, on='shop_id', how='left')
    train_df = train_df.merge(item_df, on='item_id', how='left')
    train_df = train_df.merge(item_cat_df, on='item_category_id', how='left')

    test_df = test_df.merge(shops_df, on='shop_id', how='left')
    test_df = test_df.merge(item_df, on='item_id', how='left')
    test_df = test_df.merge(item_cat_df, on='item_category_id', how='left')




if __name__ == '__main__':
    main()




