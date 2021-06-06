
from itertools import product
from os import path
from datetime import date
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from src.feature_engeering import *

BASE_PATH = '/Users/deepudilip/ML/Coursera/HowToWinDSCompeition/KagglePredictFutureSales'
DATA_PATH = path.join(BASE_PATH, 'data/')
PROCESSED_DATA_PATH = path.join(DATA_PATH, 'prepared_data/')
RESTART_PATH = path.join(DATA_PATH, 'restarts/')


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
    item_cat_df['item_sub_type_id'] = LabelEncoder().fit_transform(item_cat_df['item_sub_type'])
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
    item_df['item_id'] = item_df['item_id'].astype(np.int16)

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
    :param train_months:
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


def clean_and_write_prepared_data(data):
    """
    Convenience method to write pre-processed train and test data to disk, with appropriate names
    :param data:
    :return:
    """
    data = fill_null_for_lag_cols(data)

    # train data
    train_data = data[(data['date_block_num'] > 11) & (data['date_block_num'] < 33)]
    valid_data = data[data['date_block_num'] == 33]
    test_data = data[data['date_block_num'] == 34]

    feature_count = len(train_data.columns) - 2

    file_name_postfix = str(date.today()) + '.pkl'
    train_file_name = 'train_df_' + str(feature_count) + '_features_' + file_name_postfix
    valid_file_name = 'valid_df_' + str(feature_count) + '_features_' + file_name_postfix
    test_file_name = 'test_df_' + str(feature_count) + '_features_' + file_name_postfix

    train_data.to_pickle(path.join(PROCESSED_DATA_PATH, train_file_name))
    valid_data.to_pickle(path.join(PROCESSED_DATA_PATH, valid_file_name))
    test_data.to_pickle(path.join(PROCESSED_DATA_PATH, test_file_name))

    return True


def read_all_data(data_base_path,
                  sales_file='sales_train.csv',
                  items_file='items.csv',
                  item_categories_file='item_categories.csv',
                  shops_file='shops.csv',
                  test_file='test.csv'):

    sales_train_df = pd.read_csv(path.join(data_base_path, sales_file))
    item_df = pd.read_csv(path.join(data_base_path, items_file))
    item_cat_df = pd.read_csv(path.join(data_base_path, item_categories_file))
    shops_df = pd.read_csv(path.join(data_base_path, shops_file))
    test_df = pd.read_csv(path.join(data_base_path, test_file))

    dataframes = {'sales': sales_train_df,
                  'items': item_df,
                  'item_categories': item_cat_df,
                  'shops': shops_df,
                  'test': test_df}

    return dataframes


def merge_dataframes(sales_train_df, item_df, item_cat_df, shops_df, test_df):

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

    combined_df = pd.concat([train_df, test_df])

    return combined_df


def read_restart(step, filename='preprocessed_data'):
    print('reading intermediate dataframe after step ' + str(step))
    if step == 0:
        return pd.DataFrame()

    restart_path = path.join(RESTART_PATH, filename + '_step_' + str(step) + '.pkl')
    if not path.exists(restart_path):
        print('Error, no file found. Please check input file name or create the requested restarts')
        return pd.DataFrame()

    combined_df = pd.read_pickle(restart_path)
    return combined_df


def write_restart(data, step, filename='preprocessed_data'):
    print('writing intermediate dataframe after step ' + str(step))
    write_path = path.join(RESTART_PATH, filename + '_step_' + str(step) + '.pkl')
    data.to_pickle(write_path)
    return


def precheck_for_run_step(data, run_step_no, read_step):

    if data.empty:
        print('Error: No data to perform requested step. Exiting')
        return False
    if run_step_no <= read_step:
        print('Error: Trying to repeat previously run step. Exiting')
        return False
    return True


def fill_null_for_lag_cols(data):

    lagged_columns_with_null = [col for col in data.columns
                                if 'lagged' in col and data[col].isnull().any()]
    for col in lagged_columns_with_null:
        data[col].fillna(0, inplace=True)

    return data


def main(run_config):
    """
    Main method to run the preprocessing steps.
    :return:
    """
    read_step = run_config.get('read_step', 0)

    run_step_1 = run_config.get('run_step_1', True)
    run_step_2 = run_config.get('run_step_2', True)
    run_step_3 = run_config.get('run_step_3', True)
    run_step_4 = run_config.get('run_step_4', True)
    run_step_5 = run_config.get('run_step_5', True)
    run_step_6 = run_config.get('run_step_6', True)

    write_step_1 = run_config.get('write_step_1', False)
    write_step_2 = run_config.get('write_step_2', False)
    write_step_3 = run_config.get('write_step_3', False)
    write_step_4 = run_config.get('write_step_4', False)
    write_step_5 = run_config.get('write_step_5', False)
    write_step_6 = run_config.get('write_step_6', False)

    all_data = read_all_data(DATA_PATH)

    preprocessed_data = read_restart(read_step)

    if preprocessed_data.empty:
        run_step_1 = run_step_2 = run_step_3 = run_step_4 = run_step_5 = True

    if run_step_1:
        print('running step 1')
        sales_train_df = all_data.get('sales')
        item_df = all_data.get('items')
        item_cat_df = all_data.get('item_categories')
        shops_df = all_data.get('shops')
        test_df = all_data.get('test')

        preprocessed_data = merge_dataframes(sales_train_df, item_df, item_cat_df, shops_df, test_df)

        if write_step_1:
            write_restart(preprocessed_data, 1)

    if run_step_2:
        print('running step 2')
        if not precheck_for_run_step(preprocessed_data, 2, read_step):
            return False

        # create  mean encoded features
        print('creating mean encoded features')
        preprocessed_data = create_mean_encoded_features(preprocessed_data)

        if write_step_2:
            write_restart(preprocessed_data, 2)

    # create price trend features
    if run_step_3:
        print('running step 3')
        if not precheck_for_run_step(preprocessed_data, 3, read_step):
            return False

        print('creating price trend features')
        sales_train_df = all_data.get('sales')
        preprocessed_data = create_item_price_trend_features(preprocessed_data, sales_train_df)
        if write_step_3:
            write_restart(preprocessed_data, 3)

    # create features based on shop revenue trend
    if run_step_4:
        print('running step 4')
        if not precheck_for_run_step(preprocessed_data, 4, read_step):
            return False

        print('creating shop revenue trend features')
        sales_train_df = all_data.get('sales')
        preprocessed_data = create_shop_revenue_trend_features(preprocessed_data, sales_train_df)
        if write_step_4:
            write_restart(preprocessed_data, 4)

    # crate special features
    if run_step_5:
        print('running step 5')
        if not precheck_for_run_step(preprocessed_data, 5, read_step):
            return False

        print('creating special features')
        preprocessed_data = create_special_features(preprocessed_data)
        if write_step_5:
            write_restart(preprocessed_data, 5)

    # clean and write preprocessed data
    if run_step_6:
        print('running step 6')
        if not precheck_for_run_step(preprocessed_data, 6, read_step):
            return False

        print('cleaning and writing train, test and validation data')
        clean_and_write_prepared_data(preprocessed_data)


if __name__ == '__main__':
    run_config = {'read_step': 5,
                  'run_step_1': False,
                  'run_step_2': False,
                  'run_step_3': False,
                  'run_step_4': False,
                  'run_step_5': False,
                  'run_step_6': True,
                  'write_step_1': False,
                  'write_step_2': False,
                  'write_step_3': False,
                  'write_step_4': False,
                  'write_step_5': False}
    main(run_config)




