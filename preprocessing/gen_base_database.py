#!/usr/bin/env python
import snowflake.connector
from user_info import user, password, account

# Gets the version
ctx = snowflake.connector.connect(
    user=user,
    password=password,
    account=account
    )
cs = ctx.cursor().execute("SELECT * FROM US_STOCKS_DAILY.PUBLIC.STOCK_HISTORY WHERE symbol='AAPL' ORDER BY date")
result = cs.fetchmany(10)
for pt in result:
    print(pt)

# try:
#     cs.execute("SELECT current_version()")
#     one_row = cs.fetchone()
#     print(one_row[0])
# finally:
#     cs.close()
# ctx.close()