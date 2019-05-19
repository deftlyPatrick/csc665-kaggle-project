import pandas as pd
import numpy as np


def train_test_split(x, y, test_size, shuffle, random_state=None):

    x_temp = x.copy()
    y_temp = y.copy()

    if shuffle:
        # recombine x and y and shuffle the dataframe
        x_temp['y'] = y_temp
        x_temp = x_temp.sample(frac=1, random_state=random_state)
        y_temp = x_temp['y'].values
        x_temp = x_temp.drop('y', axis=1)

    # X split
    x_train, x_test = np.split(x_temp, [int((1 - test_size)*len(x))])
    # y split
    y_train, y_test = np.split(y_temp, [int((1 - test_size)*len(y))])

    return x_train, x_test, y_train, y_test


def create_categories (df, list_columns):

    # convert values, in-place, in the provided columns to numerical values
    for column in list_columns:
        df[column] = df[column].astype('category').cat.codes
    return


def preprocess_ver_1(csv_df, target_col_name):

    csv_df_temp = csv_df.copy()

    # remove all rows with NA values
    rows_labeled_na = csv_df_temp.isnull().any(axis=1)
    rows_with_na = csv_df_temp[rows_labeled_na]
    rows_with_data = csv_df_temp[-rows_labeled_na]
    csv_df_temp = rows_with_data

    # convert datetime to a number if a date column exists
    if 'Date' in csv_df.columns:
        csv_df_temp['Date'] = pd.to_datetime(csv_df_temp['Date'], infer_datetime_format=True)
        csv_df_temp['Date'] = csv_df_temp['Date'].astype(np.int64)

    # convert all strings to numbers
    string_columns = list(csv_df_temp.select_dtypes(exclude='number'))
    create_categories(csv_df_temp, string_columns)

    # split the data frame into x and y
    csv_df_x = csv_df_temp.drop(target_col_name, axis=1)
    csv_df_y = csv_df_temp[target_col_name].values

    return csv_df_x, csv_df_y
	
def preprocess_ver_2(csv_df, target_col_name):
	
	# replace all NA values with zero
	csv_df_temp = csv_df.copy().fillna(0)
	
	# convert datetime to a number if a date column exists
	if 'Date' in csv_df.columns:
		csv_df_temp['Date'] = pd.to_datetime(csv_df_temp['Date'], infer_datetime_format=True)
		csv_df_temp['Date'] = csv_df_temp['Date'].astype(np.int64)

	# convert all strings to numbers
	string_columns = list(csv_df_temp.select_dtypes(exclude='number'))
	create_categories(csv_df_temp, string_columns)

	# split the data frame into x and y
	csv_df_x = csv_df_temp.drop(target_col_name, axis=1)
	csv_df_y = csv_df_temp[target_col_name].values

	return csv_df_x, csv_df_y


def preprocess_ver_3(csv_df):

	# replace all NA values with zero
	csv_df_temp = csv_df.copy().fillna(0)
	
	# convert datetime to a number if a date column exists
	if 'Date' in csv_df.columns:
		csv_df_temp['Date'] = pd.to_datetime(csv_df_temp['Date'], infer_datetime_format=True)
		csv_df_temp['Date'] = csv_df_temp['Date'].astype(np.int64)

	# convert all strings to numbers
	string_columns = list(csv_df_temp.select_dtypes(exclude='number'))
	create_categories(csv_df_temp, string_columns)
	
	return csv_df_temp
	

class Object(object):
    pass
