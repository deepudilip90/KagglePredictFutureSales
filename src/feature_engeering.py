import re


def create_lagged_features(df, lags, col):
    """
    Convenience function to create lagged features (based on date_block_num).
    :param df: A dataframe on which the lagged features have to be created.
    :type df: pandas.DataFrame
    :param lags: A list of lag values for which features need to be created.
    :type lags: List
    :param col: The colum for which the lagged feature is required.
    :type col: str
    :return:
    """
    for lag in lags:
        print("now creating features for lag: ", lag)
        df_shifted = df.loc[:, ['date_block_num', 'shop_id', 'item_id', col]]
        df_shifted['date_block_num'] += lag
        df_shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_lagged_' + str(lag)]
        print('merging with original dataframe')
        df = df.merge(df_shifted, on=['date_block_num',
                                      'shop_id',
                                      'item_id'],
                      how='left')
    return df


def get_mean_encoded_features(df, group_cols, agg_col='item_cnt_month', agg_col_alias=None):
    """
    Convenience function to create a mean encoded feature based on an input set of grouping columns.
    :param df:
    :param group_cols:
    :param agg_col:
    :return:
    """
    agg_df = df.groupby(group_cols).agg({agg_col: 'mean'})
    if not agg_col_alias:
        agg_col_alias = 'mean_' + agg_col

    agg_df.columns = [agg_col_alias]
    agg_df.reset_index(inplace=True)

    df = df.merge(agg_df, on=group_cols, how='left')

    return df


def create_lagged_mean_encoded_features(df, group_cols, agg_col, lags, agg_col_alias=None):
    """
    Convenience function to create a get a set of lagged mean encoded feature based on an input set of grouping columns
    and lags.
    :todo add docu
    :param df:
    :param group_cols:
    :param agg_col:
    :param agg_col_alias:
    :return:
    """
    if not agg_col_alias:
        agg_col_alias = 'mean_' + agg_col
    df = get_mean_encoded_features(df, group_cols, agg_col, agg_col_alias)
    df = create_lagged_features(df, lags, agg_col_alias)

    return df


def create_mean_encoded_features(train_test_df):
    """
    Function to create a set of mean encoded features.
    :param train_test_df:
    :return:
    """
    # this could be parsed using an xml parser
    groups_for_mean_encode = [
                              ['date_block_num'],
                              ['date_block_num', 'item_id'],
                              ['date_block_num', 'shop_id'],
                              ['date_block_num', 'item_category_id'],
                              ['date_block_num', 'shop_id', 'item_category_id'],
                              ['date_block_num', 'shop_id', 'item_type_id'],
                              ['date_block_num', 'shop_id', 'item_sub_type_id'],
                              ['date_block_num', 'city_id'],
                              ['date_block_num', 'item_id', 'city_id'],
                              ['date_block_num', 'item_type_id'],
                              ['date_block_num', 'item_sub_type_id']
                             ]

    lags_for_mean_encode = [
                            [1],
                            [1, 2, 3, 6, 12],
                            [1, 2, 3, 6, 12],
                            [1],
                            [1],
                            [1],
                            [1],
                            [1],
                            [1],
                            [1],
                            [1]
                           ]
    for group, lags in zip(groups_for_mean_encode, lags_for_mean_encode):
        agg_col_alias = '_'.join(['_'.join([item for item in col.split('_')[0:-1]]) for col in group])
        agg_col_alias = re.sub('_block', '', agg_col_alias) + '_avg_item_cnt'
        print('adding mean encoded feature for ' + agg_col_alias + ' with lags ', ', '.join([str(lag) for lag in lags]))
        train_test_df = create_lagged_mean_encoded_features(train_test_df, group, 'item_cnt_month', lags, agg_col_alias)

    return train_test_df


def create_item_price_trend_features(train_test_df, sales_train_df):
    """
    Method to create price trend related features.
    :param sales_df:
    :return:
    """
    # average item price across entire training data
    avg_item_price = sales_train_df.groupby(['item_id']).agg({'item_price': 'mean'})
    avg_item_price.columns = ['item_avg_price']
    avg_item_price.reset_index(inplace=True)

    # average item price monthwise for training data
    avg_date_block_item_price = sales_train_df.groupby(['date_block_num', 'item_id']).agg({'item_price': 'mean'})
    avg_date_block_item_price.columns = ['date_item_avg_price']
    avg_date_block_item_price.reset_index(inplace=True)

    # merge the above to original trainig dataframe
    train_test_df = train_test_df.merge(avg_item_price, on='item_id', how='left')
    train_test_df = train_test_df.merge(avg_date_block_item_price, on=['item_id', 'date_block_num'], how='left')

    # for monthwise item price - create lagged features
    lags = [1, 2, 3, 4, 5, 6]
    train_test_df = create_lagged_features(train_test_df, lags=lags, col='date_item_avg_price')

    for lag in lags:
        train_test_df['perc_delta_price_lagged_' + str(lag)] = ((train_test_df['date_item_avg_price_lagged_' + str(lag)]
                                                                 - train_test_df['item_avg_price']) / train_test_df[
                                                                    'item_avg_price'])

    def select_trend_value(row):
        for lag in lags:
            if row['perc_delta_price_lagged_' + str(lag)]:
                return row['perc_delta_price_lagged_' + str(lag)]
        return 0

    train_test_df['perc_detla_price_lagged'] = train_test_df.apply(select_trend_value, axis=1)
    features_to_drop = (['date_item_avg_price_lagged_' + str(lag) for lag in lags]
                        + ['perc_delta_price_lagged_' + str(lag) for lag in lags]
                        + ['item_avg_price'])

    train_test_df = train_test_df.drop(columns=features_to_drop)

    return train_test_df


def create_shop_revenue_trend_features(train_test_df, sales_train_df):
    """
    :todo add docu
    :param sales_train_df:
    :param train_test_df:
    :return:
    """
    sales_train_df['revenue'] = sales_train_df['item_cnt_day'] * sales_train_df['item_price']
    shop_revenue_df = sales_train_df.groupby(['date_block_num',
                                              'shop_id']).agg({'revenue': 'sum'})
    shop_revenue_df.columns = ['date_shop_total_revenue']
    shop_revenue_df.reset_index(inplace=True)

    # shop average revenue across all months
    shop_revenue_df['shop_avg_revenue'] = (shop_revenue_df.groupby(['shop_id'])['date_shop_total_revenue'].
                                           transform('mean'))

    shop_revenue_df['delta_revenue'] = (shop_revenue_df['date_shop_total_revenue']
                                        - shop_revenue_df['shop_avg_revenue']) / shop_revenue_df['shop_avg_revenue']
    train_test_df = train_test_df.merge(shop_revenue_df, on=['date_block_num', 'shop_id'], how='left')
    train_test_df = create_lagged_features(train_test_df, [1], 'delta_revenue')
    train_test_df = train_test_df.drop(columns=['delta_revenue', 'date_shop_total_revenue', 'shop_avg_revenue'])

    return train_test_df

# :todo: check how many nulls in shop revenue trend feature
# compare results

def create_special_features(train_test_df):
    """
    :todo add docu
    :param train_test_df:
    :return:
    """
    # Months since last sale for item - shop combination
    shop_item_ordered_train_df = train_test_df.sort_values(by=['shop_id', 'item_id', 'date_block_num'])
    shop_item_ordered_train_df['item_shop_last_sale_date'] = (shop_item_ordered_train_df
                                                              .groupby(['shop_id', 'item_id'])['date_block_num']
                                                              .shift(1))
    shop_item_ordered_train_df['mnths_since_last'] = (shop_item_ordered_train_df['date_block_num'] -
                                                      shop_item_ordered_train_df['item_shop_last_sale_date'])
    shop_item_ordered_train_df['mnths_since_last'] = (shop_item_ordered_train_df['date_block_num'] -
                                                      shop_item_ordered_train_df['item_shop_last_sale_date'])

    shop_item_ordered_train_df['mnths_since_last'].fillna(-1)

    train_test_df = shop_item_ordered_train_df.sort_values(by=['date_block_num', 'shop_id', 'item_id'])

    # Months since first sale for shop-item combination

    train_test_df['item_shop_first_sale_month'] = (train_test_df.groupby(['shop_id', 'item_id'])['date_block_num']
                                                   .transform('min'))
    train_test_df['item_shop_mnths_since_first'] = (train_test_df['date_block_num']
                                                    - train_test_df['item_shop_first_sale_month'])

    train_test_df['item_first_sale_month'] = train_test_df.groupby(['item_id'])['date_block_num'].transform('min')
    train_test_df['item_mnths_since_first'] = train_test_df['date_block_num'] - train_test_df['item_first_sale_month']

    train_test_df = train_test_df.drop(columns=['item_shop_first_sale_month', 'item_first_sale_month'])

    return train_test_df






