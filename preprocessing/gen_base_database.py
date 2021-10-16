#!/usr/bin/env python
import snowflake.connector
from user_info import user, password, account
import pandas as pd
import tqdm
#must create user_info with user, password, account value yourself

# Gets the version
ctx = snowflake.connector.connect(
    user=user,
    password=password,
    account=account
    )

def year_difference(year: int) -> int:
    query_string = f"SELECT * FROM US_STOCKS_DAILY.PUBLIC.STOCK_HISTORY WHERE symbol='AAPL' AND date >= '{year}-01-01' AND date < '{year+1}-1-1' ORDER BY date"
    data_cursor = ctx.cursor().execute(query_string)
    data = data_cursor.fetchall()

    year_start = data[0][3] #opening stock price for the year
    year_end = data[-1][6] #closing stock price for the year
    return year_start - year_end

def get_inc_dec(diff:int) -> int:
    """
    converts yearly difference into 1 (for annual increase) and 0 (for annual decrease)
    """
    if (diff <= 0):
        return 0
    else:
        return 1

def get_info_list(ticker:str, low_year:int, high_year:int) -> list:
    """
    gets info between low_year and high year inclusive
    """
    info_cursor = ctx.cursor().execute(f"SELECT * FROM US_STOCKS_DAILY.PUBLIC.STOCK_HISTORY WHERE symbol='{ticker}' AND date<'{high_year+1}-1-1' AND date>='{low_year}-1-1' ORDER BY date")
    return info_cursor.fetchall()

def get_feature_vector(point:tuple) -> list:
    l = []

    l.append(point[3])
    l.append(point[4])
    l.append(point[5])
    l.append(point[6])
    l.append(point[7])
    l.append(point[8])

    return l

def main():
    prediction_year = 2019
    train_range = (2008, 2018)
    datalist = []
    # tickerList = []
    ticker = 'AAPL'
    
    infoList = get_info_list(ticker, train_range[0], train_range[1])
    target = get_inc_dec(year_difference(prediction_year))
    print(len(infoList))

    infoList = infoList[0:-1:10]

    for pnt in infoList:
        d = dict()
        
        

    


if __name__=="__main__":
    main()