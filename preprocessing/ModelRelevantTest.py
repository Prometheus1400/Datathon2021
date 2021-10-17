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
        f"""
SELECT Stock_history."SYMBOL", Stock_history."Year", Stock_history."Month", Stock_history."OPEN", 
    Stock_history."HIGH", Stock_history."LOW", Stock_history."CLOSE", 
    Stock_history."VOLUME", Stock_history."ADJCLOSE",
    Env_var."CO2 from Oil", Env_var."CO2 from Coal",
    Env_var."CO2 from Gas", Env_var."Total CO2",
    Irrelevant."Value"
FROM
    ((SELECT 
        SYMBOL, YEAR(DATE) as "Year", MONTH(DATE) as "Month",OPEN, HIGH, LOW, CLOSE, VOLUME, ADJCLOSE    
    FROM
        (SELECT *,row_number() OVER (ORDER BY DATE) AS rownum 
         FROM US_STOCKS_DAILY.PUBLIC.STOCK_HISTORY 
         WHERE SYMBOL='{ticker}' AND DATE<'2020-01-01' AND DATE >'2008-12-31' 
         ORDER BY DATE) 
        AS All_history
    WHERE All_history.rownum%10=1) AS Stock_history)
INNER JOIN
    ((select Year(Oil."Date") as "Year", Month(Oil."Date") as "Month", Oil."Value" as "CO2 from Oil", Coal."Value" as "CO2 from Coal", Gas."Value" as "CO2 from Gas", Total."Value" as "Total CO2"   from
    ((SELECT * FROM "ENVIRONMENT_DATA_ATLAS"."ENVIRONMENT"."CDIACCO2STATES" WHERE "Variable Name"='Carbon Emissions from Oil' and "State regionid" = 'US' and "Frequency" = 'M' order by "Date") as Oil)
    inner join
    ((SELECT * FROM "ENVIRONMENT_DATA_ATLAS"."ENVIRONMENT"."CDIACCO2STATES" WHERE "Variable Name"='Carbon Emissions from Coal' and "State regionid" = 'US' and "Frequency" = 'M' order by "Date") as Coal)
        on Oil."Date" = Coal."Date"
    inner join
    ((SELECT * FROM "ENVIRONMENT_DATA_ATLAS"."ENVIRONMENT"."CDIACCO2STATES" WHERE "Variable Name"='Carbon Emissions from Gas' and "State regionid" = 'US' and "Frequency" = 'M' order by "Date") as Gas)
        on Oil."Date" = Gas."Date"
    inner join
    ((SELECT * FROM "ENVIRONMENT_DATA_ATLAS"."ENVIRONMENT"."CDIACCO2STATES" WHERE "Variable Name"='Total Carbon Emissions' and "State regionid" = 'US' and "Frequency" = 'M' order by "Date") as Total)
        on Oil."Date" = Total."Date") as Env_var)
ON
    Env_var."Month" = Stock_history."Month" and Env_var."Year" = Stock_history."Year"
INNER JOIN
    ((select MONTH("Date") as "Month", YEAR("Date") as "Year", "Value" from "ENVIRONMENT_DATA_ATLAS"."ENVIRONMENT"."WRISGWL2019" where "Location Name" = 'New Delhi' and "Frequency" = 'M' and "Indicator Name" = 'Rainfall Actual Level' order by "Date") as Irrelevant)
ON Stock_history."Month" = Irrelevant."Month" and Stock_history."Year" = Irrelevant."Year";
"""
    )
    # info_cursor = ctx.cursor().execute(
    #     f"SELECT * FROM US_STOCKS_DAILY.PUBLIC.STOCK_HISTORY WHERE symbol='{ticker}' AND date<'{high_year+1}-1-1' AND date>='{low_year}-1-1' ORDER BY date")
    return info_cursor.fetchall()


def get_feature_vector(point: tuple) -> list:
    l = []

    l.append(point[3])
    l.append(point[4])
    l.append(point[5])
    l.append(point[6])
    l.append(point[7])
    l.append(point[8])

    return l


# def getFeatureVector(curr: tuple) -> list:
#     global yearDiff
#     l = []
#     for i in range(3, 9):
#         l.append(curr[i])
#     l = list(map(float, l))
#     trend = -1
#     if (curr[1], curr[2].year) not in yearDiff:
#         trend = year_difference(curr[1], curr[2].year)
#         yearDiff[curr[1], curr[2].year] = trend
#     else:
#         trend = yearDiff[curr[1], curr[2].year]
#     convertToClass = get_inc_dec(trend)
#     l.append(convertToClass)
#     return l

def getFeatureVector(curr: tuple) -> list:
    global yearDiff
    l = []
    for i in range(1, len(curr)):
        l.append(curr[i])
    trend = -1
    if (curr[0], curr[1]) not in yearDiff:
        trend = year_difference(curr[0], curr[1])
        yearDiff[curr[0], curr[1]] = trend
    else:
        trend = yearDiff[curr[0], curr[1]]
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
    # print(label.shape)
    # print(x.shape)
    # train_X, test_X, train_Y, test_Y = train_test_split(
    #     x, label, test_size=0.3, random_state=0)
    # SVM = svm.SVC(C=100, kernel='rbf')
    currKfold = KFold(n_splits=4, shuffle=True, random_state=0)
    # # result = cross_val_score(SVM, x, label, cv=currKfold)
    # # print("Avg accuracy: {}".format(result.mean()))
    # result = cross_validate(SVM, X=x, y=label, scoring=[
    #     'accuracy', 'recall', 'precision'], cv=currKfold)
    # avg_recall = np.mean(result['test_recall'])
    # avg_accuracy = np.mean(result['test_accuracy'])
    # avg_precision = np.mean(result['test_precision'])
    # print("Ticker: ", ticker)
    # print("Recall: ", avg_recall)
    # print("Accuracy: ", avg_accuracy)
    # print("Precision: ", avg_precision)
    # svclassifier = svm.SVC(C=100, kernel='rbf')
    # svclassifier.fit(x, label)
    # filename = 'BaselineModel.sav'
    # joblib.dump(svclassifier, filename)
    # pred_Y = svclassifier.predict(test_X)
    # print(confusion_matrix(test_Y, pred_Y))
    # print(classification_report(test_Y, pred_Y))
    loaded_model = joblib.load("./trained_models/Relevant.sav")
    result = cross_validate(loaded_model, X=x, y=label, scoring=[
        'accuracy', 'recall', 'precision'], cv=currKfold)
    avg_recall = np.mean(result['test_recall'])
    avg_accuracy = np.mean(result['test_accuracy'])
    avg_precision = np.mean(result['test_precision'])
    print("With additional features but relevant such as CO2 emission and rainfall in India (one of the big importer for oil):")
    print("Ticker: ", ticker)
    print("Recall: ", avg_recall)
    print("Accuracy: ", avg_accuracy)
    print("Precision: ", avg_precision)
    print()
    return result


def main():
    # prediction_year = 2019
    # train_range = (2008, 2018)
    # datalist = []
    # # tickerList = []
    # ticker = 'AAPL'

    # # 1 list with tuple as 1 data point
    # infoList = get_info_list(ticker, train_range[0], train_range[1])
    # target = get_inc_dec(year_difference(ticker, prediction_year))

    # infoList = infoList[0:-1:10]  # filter by every 10 data points
    # for line in infoList:
    #     datalist.append(get_feature_vector(line))  # 1 list of list

    # d = dict()
    # d['ticker'] = ticker
    # d['x'] = datalist
    # d['y'] = target

    # Energy, Technology, Healthcare, Consumer centered
    test_data = ['PUMP', 'TOT', 'PNRL', 'TXN', 'FLEX',
                 'LORL', 'CVS', 'RCKT', 'CPSI', 'WMT', 'GPI', 'KBSF']
    for ticker in test_data:
        dataList = generateTicker(ticker, 2009, 2018)
        # print(dataList)
        train_model(ticker, dataList)

    # with open("ticker.txt") as file:
    #     lines = file.readlines()
    #     lines = [line.rstrip() for line in lines]
    # ticker = lines
    # data = []
    # for i in range(1):
    #     dataList = generateTicker(ticker[i], 2009, 2018)
    #     data += dataList
    # # print(data[0])
    # # print(len(data))
    # train_model("", data)


if __name__ == "__main__":
    main()