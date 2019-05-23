import re
import pandas as pd
import numpy as np
from typing import Tuple, AnyStr
from pandas.api.types import is_string_dtype
#-----------------------------------------------------------------------------------------
def create_categories(df: pd.DataFrame):
    for name, col in df.items():
        if is_string_dtype(col):
            df[name] = df[name].astype('category').cat.codes
#-----------------------------------------------------------------------------------------
def preprocess_ver_1(csv_df: pd.DataFrame, target_col_name: AnyStr) -> Tuple[
        pd.DataFrame, np.ndarray]:
    """
        Returns X, y
    """
    rows_labeled_na = csv_df.isnull().any(axis=1)
    rows_with_data = csv_df[~rows_labeled_na].copy()

    rows_with_data['Date'] = pd.to_datetime(rows_with_data['Date'],
                                            infer_datetime_format=True)
    rows_with_data['Date'] = rows_with_data['Date'].astype(np.int64)

    create_categories(rows_with_data)

    feat_df = rows_with_data.drop(target_col_name, axis=1)
    X = feat_df
    y = rows_with_data[target_col_name].values

    return X, y
#-----------------------------------------------------------------------------------------
def train_test_split(X: pd.DataFrame, y: np.array, test_size, shuffle, random_state=None) \
        -> Tuple[pd.DataFrame, pd.DataFrame, np.array, np.array]:
    """
        Returns (X_train, X_test, y_train, y_test)
    """
    if shuffle:
        rs = np.random.RandomState(random_state)
        shuffled_indices = rs.permutation(X.shape[0])
        X_shuffled = X.iloc[shuffled_indices]
        y_shuffled = y[shuffled_indices]
    else:
        X_shuffled = X
        y_shuffled = y

    train_end = int(X.shape[0] * (1 - test_size))
    assert train_end < X.shape[0]  # Make sure something is left for the test set

    return X_shuffled[:train_end], X_shuffled[train_end:], \
           y_shuffled[:train_end], y_shuffled[train_end:]
#-----------------------------------------------------------------------------------------
def create_datetime_features(df: pd.DataFrame,
                             field_name: str,
                             drop_orig: bool = True,
                             add_time: bool = False):
    field = df[field_name]
    assert isinstance(field, pd.Series)

    field_type = field.dtype

    if not np.issubdtype(field_type, np.datetime64):
        df[field_name] = field = pd.to_datetime(field, infer_datetime_format=True)
        assert isinstance(field, pd.Series)

    col_prefix = re.sub('[Dd]ate$', '', field_name)

    attr = ['year', 'month', 'week', 'day', 'dayofweek', 'dayofyear',
            'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
            'is_year_start', 'is_year_end']

    if add_time:
        attr += ['hour', 'minute', 'second']

    for name in attr:
        df[col_prefix + '_' + name] = getattr(field.dt, name.lower())

    df[col_prefix + "_elapsed"] = field.astype(np.int64)  # Seconds ?

    if drop_orig:
        df.drop([field_name], axis=1, inplace=True)
#-----------------------------------------------------------------------------------------
class Object(object):
    pass
