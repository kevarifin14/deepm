import numpy as np
import pandas as pd
from constants import *
import sqlite3


class MarketHistory:
    def __init__(self, start, end, features=['date', 'high', 'low', 'close']):
        self.coins = ['BCH', 'DASH','DGB' , 'ETC', 'ETH', 'FCT', 'GNT', 'LTC', 'SC', 'STR', 'XEM', 'XMR', 'XRP', 'ZEC', 'reversed_USDT']                
        self.features = ['close', 'high', 'low']
        self.__storage_period = 300
        self.data = self.get_global_data_matrix(start, end)

    def get_global_data_matrix(self, start, end, period=1800):
        return self.get_global_panel(start, end, period).values

    def get_global_panel(self, start, end, period=1800):
        start = int(start - (start%period))
        end = int(end - (end%period))
        
        coins = self.coins
        features = self.features

        # logging.info("feature type list is %s" % str(features))
        # self.__checkperiod(period)

        time_index = pd.to_datetime(list(range(start, end+1, period)),unit='s')
        panel = pd.Panel(items=features, major_axis=coins, minor_axis=time_index, dtype=np.float32)

        connection = sqlite3.connect(DATABASE_DIR)
        try:
            for row_number, coin in enumerate(coins):
                for feature in features:
                    # NOTE: transform the start date to end date
                    if feature == "close":
                        sql = ("SELECT date+300 AS date_norm, close FROM History WHERE"
                               " date_norm>={start} and date_norm<={end}" 
                               " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                               start=start, end=end, period=period, coin=coin))
                    elif feature == "open":
                        sql = ("SELECT date+{period} AS date_norm, open FROM History WHERE"
                               " date_norm>={start} and date_norm<={end}" 
                               " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                               start=start, end=end, period=period, coin=coin))
                    elif feature == "volume":
                        sql = ("SELECT date_norm, SUM(volume)"+
                               " FROM (SELECT date+{period}-(date%{period}) "
                               "AS date_norm, volume, coin FROM History)"
                               " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                               " GROUP BY date_norm".format(
                                    period=period,start=start,end=end,coin=coin))
                    elif feature == "high":
                        sql = ("SELECT date_norm, MAX(high)" +
                               " FROM (SELECT date+{period}-(date%{period})"
                               " AS date_norm, high, coin FROM History)"
                               " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                               " GROUP BY date_norm".format(
                                    period=period,start=start,end=end,coin=coin))
                    elif feature == "low":
                        sql = ("SELECT date_norm, MIN(low)" +
                                " FROM (SELECT date+{period}-(date%{period})"
                                " AS date_norm, low, coin FROM History)"
                                " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                                " GROUP BY date_norm".format(
                                    period=period,start=start,end=end,coin=coin))
                    else:
                        msg = ("The feature %s is not supported" % feature)
                        raise ValueError(msg)
                    serial_data = pd.read_sql_query(sql, con=connection,
                                                    parse_dates=["date_norm"],
                                                    index_col="date_norm")
                    panel.loc[feature, coin, serial_data.index] = serial_data.squeeze()
                    panel = self.panel_fillna(panel, "both")
        finally:
            connection.commit()
            connection.close()
        return panel

    def panel_fillna(self, panel, type="bfill"):
        frames = {}
        for item in panel.items:
            if type == "both":
                frames[item] = panel.loc[item].fillna(axis=1, method="bfill").\
                    fillna(axis=1, method="ffill")
            else:
                frames[item] = panel.loc[item].fillna(axis=1, method=type)
        return pd.Panel(frames)

