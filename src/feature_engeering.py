

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




