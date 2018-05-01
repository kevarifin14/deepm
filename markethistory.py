import numpy as np
import pandas as pd
import sqlite3
import time
import matplotlib.pyplot as plt

from constants import *
from datetime import datetime


class MarketHistory:
    def __init__(self, config):
        self.config = config['market_history']
        self.coins = self.config['coins']
        self.features = self.config['features']
        self.short_window = self.config['short_window']
        self.long_window = self.config['long_window']  
        self.moving_average = self.config['moving_average']      
        start_unix = int(self.parse_time(self.config['start_date']))
        end_unix = int(self.parse_time(self.config['end_date']))
        self.data = self.get_global_data_matrix(start_unix, end_unix)

    def get_global_data_matrix(self, start, end, period=1800):
        matrix = self.get_global_panel(start, end, period).values
        if self.moving_average:
            df = pd.DataFrame(matrix[0])
            ma_short = df.rolling(window=self.short_window, axis=1).mean().as_matrix()[None, :, :]
            ma_long = df.rolling(window=self.long_window, axis=1).mean().as_matrix()[None, :, :]
            matrix = np.vstack((matrix, ma_short))
            matrix = np.vstack((matrix, ma_long))
        return self.matrix_filter_missing_coins(matrix)

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
    
    def matrix_filter_missing_coins(self, panel):
        ###Check that data is present for all coins:
        to_delete = []
        for i in range(panel.shape[1]):
            if np.isnan(panel[:,i,:]).sum() > (panel.shape[0] * panel.shape[2])/2:
                to_delete.append(i)
        panel = np.delete(panel,to_delete,axis=1)
        bad_coins = [self.coins[i] for i in range(len(self.coins)) if i in to_delete]
        self.traded_coins = [self.coins[i] for i in range(len(self.coins)) if i not in to_delete]
        print("Warning: missing data for following coins " + str(bad_coins))
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
    
    def parse_time(self,time_string):
        return time.mktime(datetime.strptime(time_string, "%Y/%m/%d").timetuple())
