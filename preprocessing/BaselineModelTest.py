#!/usr/bin/env python
import snowflake.connector
from snowflake.connector.cursor import CAN_USE_ARROW_RESULT_FORMAT
from user_info import user, password, account

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn import svm
from sklearn.metrics import recall_score, accuracy_score, classification_report, confusion_matrix

import joblib
import pandas as pd
import numpy as np
import tqdm
# must create user_info with user, password, account value yourself

# Gets the version
ctx = snowflake.connector.connect(
    user=user,
    password=password,
    account=account
)

yearDiff = dict()


def year_difference(ticker: str, year: int) -> int:
    query_string = f"SELECT * FROM US_STOCKS_DAILY.PUBLIC.STOCK_HISTORY WHERE symbol='{ticker}' AND date >= '{year}-01-01' AND date < '{year+1}-1-1' ORDER BY date"
    data_cursor = ctx.cursor().execute(query_string)
    data = data_cursor.fetchall()

    year_start = data[0][3]  # opening stock price for the year
    year_end = data[-1][6]  # closing stock price for the year
    return year_end - year_start


def get_inc_dec(diff: int) -> int:
    """
    converts yearly difference into 1 (for annual increase) and 0 (for annual decrease)
    """
    if (diff <= 0):
        return 0
    else:
        return 1


def get_info_list(ticker: str, low_year: int, high_year: int) -> list:
    """
    gets info between low_year and high year inclusive
    """
    info_cursor = ctx.cursor().execute(
        f"SELECT * FROM US_STOCKS_DAILY.PUBLIC.STOCK_HISTORY WHERE symbol='{ticker}' AND date<'{high_year+1}-1-1' AND date>='{low_year}-1-1' ORDER BY date")
    return info_cursor.fetchall()


def getFeatureVector(curr: tuple) -> list:
    global yearDiff
    l = []
    for i in range(3, 9):
        l.append(curr[i])
    l = list(map(float, l))
    trend = -1
    if (curr[1], curr[2].year) not in yearDiff:
        trend = year_difference(curr[1], curr[2].year)
        yearDiff[curr[1], curr[2].year] = trend
    else:
        trend = yearDiff[curr[1], curr[2].year]
    convertToClass = get_inc_dec(trend)
    l.append(convertToClass)
    return l


def generateTicker(ticker: str, start_year: int, end_year: int) -> list:
    train_range = (start_year, end_year)
    dataList = []
    infoList = get_info_list(ticker, train_range[0], train_range[1])
    infoList = infoList[0:-1:10]
    for obj in infoList:
        dataList.append(getFeatureVector(obj))
    return dataList


def train_model(ticker: str, data: list) -> list:
    result = []
    dataset = np.array(data)
    label = dataset[:, -1].astype(int)
    x = dataset[:, :-1]
    currKfold = KFold(n_splits=4, shuffle=True, random_state=0)
    loaded_model = joblib.load("./trained_models/BaselineModel.sav")
    result = cross_validate(loaded_model, X=x, y=label, scoring=[
        'accuracy', 'recall', 'precision'], cv=currKfold)
    avg_recall = np.mean(result['test_recall'])
    avg_accuracy = np.mean(result['test_accuracy'])
    avg_precision = np.mean(result['test_precision'])
    print("With no additional environmental features:")
    print("Ticker: ", ticker)
    print("Recall: ", avg_recall)
    print("Accuracy: ", avg_accuracy)
    print("Precision: ", avg_precision)
    print()
    return result


def main():
    # Energy, Technology, Healthcare, Consumer centered
    test_data = ['PUMP', 'TOT', 'PNRL', 'TXN', 'FLEX',
                 'LORL', 'CVS', 'RCKT', 'CPSI', 'WMT', 'GPI', 'KBSF']
    for ticker in test_data:
        dataList = generateTicker(ticker, 2009, 2018)
        # print(dataList)
        train_model(ticker, dataList)


if __name__ == "__main__":
    main()
