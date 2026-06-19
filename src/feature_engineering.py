import pandas as pd
import numpy as np

def extract_time_features(df, date_col='date'):
    """
    Extract temporal features from datetime column.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['week'] = df[date_col].dt.isocalendar().week.astype(int)
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df

def add_lags(df, target_col='sales', group_cols=['store', 'item'], lags=[7, 14, 30]):
    """
    Generate lag features grouped by store and item.
    """
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
    return df

def add_rolling_means(df, target_col='sales', group_cols=['store', 'item'], windows=[7, 14, 30]):
    """
    Generate rolling mean features grouped by store and item.
    Shift by 1 is applied first to prevent lookahead leakage.
    """
    df = df.copy()
    for window in windows:
        df[f'{target_col}_roll_mean_{window}'] = df.groupby(group_cols)[target_col] \
                                                  .shift(1) \
                                                  .transform(lambda x: x.rolling(window=window).mean())
    return df

def build_features(df, date_col='date', target_col='sales', group_cols=['store', 'item']):
    """
    Execute full feature engineering pipeline.
    """
    df_sorted = df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    df_time = extract_time_features(df_sorted, date_col)
    df_lags = add_lags(df_time, target_col, group_cols)
    df_features = add_rolling_means(df_lags, target_col, group_cols)
    return df_features
