import pandas as pd
import os

FILENAME = "./data/daily-min-temperatures.csv"


def load_dataset(path):
    series = pd.read_csv(path, header=0, index_col=0, parse_dates=True, squeeze=True)
    dataframe = pd.DataFrame()
    dataframe["month"] = [series.index[i].month for i in range(len(series))]
    dataframe["day"] = [series.index[i].day for i in range(len(series))]
    dataframe["temperature"] = [series[i] for i in range(len(series))]
    return dataframe


def shift_and_concat(n_shifts, column):
    """ shift the univariate time series by n_shifts and create addition columns in pandas to accomodate for shift"""
    dataframe = load_dataset(FILENAME)
    ts_values = dataframe[column]

    group_ts_series = pd.Series()
    lag_col_names = []

    for i in range(n_shifts):
        lag_name = f"t-{i+1}"
        lag_col_names.append(lag_name)
        lagged_series = pd.Series(ts_values.shift(i), name=lag_name)
        group_ts_series = pd.concat([lagged_series, group_ts_series], axis=1)

    # a column called 0 is made in an empty dataframe that needs to be dropped before combining
    group_ts_series.drop(0, axis=1, inplace=True)
    dataframe = pd.concat([group_ts_series, ts_values], axis=1)
    dataframe.columns = sorted(lag_col_names, reverse=True) + ["t"]
    return dataframe


def agg_window_statistics(dataframe, width=0, column="temperature", win_type=None):
    if win_type == "rolling":
        ts_values = dataframe[column]
        window = ts_values.shift(width - 1)
        agg_stat_df = pd.concat(
            [window.mean(), window.min(), window.max(), ts_values], axis=1
        )
        agg_stat_df.columns = ["mean", "min", "max", "t"]
    elif win_type == "expanding":
        """ if expanding window, then ignore the width argument """
        ts_values = dataframe[column]
        window = ts_values.expanding()
        agg_stat_df = pd.concat(
            [window.mean(), window.min(), window.max(), ts_values.shift(-1)], axis=1
        )
        agg_stat_df.columns = ["mean", "min", "max", "t"]
    return agg_stat_df

