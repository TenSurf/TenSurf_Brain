import os
import csv
import pytz
import re
import pandas as pd
import random
from datetime import timedelta, datetime
from time import sleep
import time
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import argrelextrema
import bisect
import numpy as np
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from influxdb_client import InfluxDBClient, Point, WriteOptions, WritePrecision


def convert_datetime(dt):
    if type(dt) == type(""):
        dt = datetime(dt)
    us_timezone = pytz.timezone('US/Pacific')
    local_time = dt.replace(tzinfo=us_timezone)
    utc_time = local_time.astimezone(pytz.utc)
    return utc_time.strftime('%Y-%m-%dT%H:%M:%SZ')


class InfluxClient:
    def __init__(self, token, org, url, bucket):
        self.token = token
        self.org = org
        self.url = url
        self.bucket = bucket
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org, timeout=None)
        self.receiver_email = 'tensurfllc@gmail.com'
        self.last_error_time = 0
        self.column_convertor = {'Candle_ClosePrice': 'close', 'Candle_HighPrice': 'high', 'BarPeriod': 'end_time',
                                 'Candle_OpenPrice': 'open', 'Candle_LowPrice': 'low', 'VP': 'volume_profile',
                                 'Candle_AskNumberOfTrades': 'ask_number', 'Candle_AskVolumeOfTrades': 'ask_volume',
                                 'Candle_BidNumberOfTrades': 'bid_number', 'Candle_BidVolumeOfTrades': 'bid_volume',
                                 'Candle_LastTradePrice': 'lastprice', 'Candle_NumberOfTrades': 'numtrades',
                                 'Candle_VolumeofTrades': 'volume', 'VAP_AskVolumes': 'vap_askvolumes',
                                 'VAP_BidVolumes': 'vap_bidvolumes', 'VAP_NumberOfTrades': 'vap_numberoftrades',
                                 'VAP_TotalVolume': 'vap_totalvolume', 'VAP_Volumes': 'vap_volumes',
                                 'VAP_Prices': 'vap_prices', "LOB_SumBid": 'sum_lob_bid', "LOB_SumAsk": 'sum_lob_ask',
                                 "LOB_SumBidTick": 'sum_lob_bid_tick', "LOB_SumAskTick": 'sum_lob_ask_tick',
                                 "LOB_BidPrices": 'lob_bid_price', "LOB_BidVolumes": 'lob_bid_volume',
                                 "LOB_AskPrices": 'lob_ask_price', "LOB_AskVolumes": 'lob_ask_volume'}

         #self.get_last_records = memory.cache(self.get_last_records)

    def retrieve_db_df_between(self, symbol, start_date, end_date):
        start_date = convert_datetime(start_date)
        end_date = convert_datetime(end_date)

        query = f'''from(bucket:"{self.bucket}") 
                |> range(start: time(v:{start_date}), stop: time(v: {end_date}))
                |> filter(fn: (r) => r["_measurement"] == "{symbol}" and\
                    r["timeframe"] == "1min" and\
                    r["liq_threshold"] == "-1")
                |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
                '''
        return self.read_data(query)

    def read_data(self, query):
        try:
            result_ = self.client.query_api().query(query=query)
            records = []
            for table in result_:
                for record in table.records:
                    records.append(record.values)

            result = pd.DataFrame(records)
            if isinstance(result, pd.DataFrame) and all(col in result.columns for col in
                                                        ['_time', 'result', 'table', '_start', '_stop', '_measurement',
                                                         'liq_threshold', 'timeframe']):
                df = result.drop(columns={'_time', 'result', 'table', '_start', '_stop', '_measurement', 'liq_threshold',
                             'timeframe'})
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                if 'BarPeriod' in df.columns:
                    df['BarPeriod'] = pd.to_datetime(df['BarPeriod'])
                self.client.close()
                df = df.rename(columns={k: v for k, v in self.column_convertor.items() if k in df.columns})
                return df
            else:
                print(query)
                print('[Error] result output is not DataFrame')
        except Exception as e:
                #self.send_error_email(f'InfluxDB - Error reading data from InfluxDB', e)
            print(f"Error reading data from InfluxDB: {e}\n Query is {query}")


class SRDetector:
    def __init__(self, pre_data, timeframe, params_, detection_method_):
        # Note: pre_data is time indexed dataframe
        self.detection_method = detection_method_
        self.timeframe = timeframe
        self.params = params_
        self.timeframe2 = re.sub(r'\d+', '1', timeframe)
        self.merge_percent = self.params[self.detection_method]['merge_percent'][self.timeframe2]
        self.closeness = self.params[self.detection_method]['merge_percent'][self.timeframe2]
        self.score_method = self.params[self.detection_method]['score']
        self.window_size_score = self.params[self.detection_method]['window_size'][self.timeframe2]
        self.last_date = None
        self.recent_candle = {'time': None, 'open': 0, 'high': 0, 'low': 0, 'close': 0}
        self.potential_support_levels = []
        self.potential_resistance_levels = []
        self.new_potential_level = False
        self.remove_level = False
        self.recent_zones = []
        self.lines = {}
        self.all_levels = []
        self.last_support_index = 0
        self.last_resistance_index = 0
        self.distance_max_threshold = 5
        self.current_round_date = None
        self.resample(pre_data, self.timeframe)

    def resample(self, data: pd.DataFrame, timeframe):
        if timeframe is None:
            return data
        tmp = data.resample(timeframe).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        tmp = tmp.dropna(subset=['open'])
        tmp['tr'] = np.maximum.reduce([tmp['high'] - tmp['low'], np.abs(tmp['high'] - tmp['close']), np.abs(tmp['low'] - tmp['close'])])
        tmp['atr'] = tmp['tr'].rolling(14).mean()
        tmp = tmp.iloc[14:]
        self.data = tmp.to_dict(orient='list')
        self.data['time'] = tmp.index.tolist()

    def get_levels(self):
        for i in range(len(self.data['time'])):
            self.atr = self.data['atr'][i]
            self.check_cross_line(i)
            self.update_potential_levels(i)
            self.score_lines(i)

    def get_distance(self):
        return self.merge_percent * self.atr

    def check_cross_line(self, index):
        close = self.data['close'][index]
        high = self.data['high'][index]
        low = self.data['low'][index]

        endtime = self.data['time'][index]
        max_cross = self.params[self.detection_method]['max_cross']
        removed_levels = []
        self.remove_level = False
        for level in self.lines.keys():
            if high >= level and not self.lines[level]['isSupport']:
                self.lines[level]['cross'] += 1
                self.lines[level]['isSupport'] = True
                if self.lines[level]['cross'] >= max_cross:
                    self.lines[level]["endTime"] = endtime
                    removed_levels.append(level)
                    if level in self.all_levels:
                        self.all_levels.remove(level)
            if low <= level and self.lines[level]['isSupport']:
                self.lines[level]['cross'] += 1
                self.lines[level]['isSupport'] = False
                if self.lines[level]['cross'] >= max_cross:
                    self.lines[level]["endTime"] = endtime
                    if level in self.all_levels:
                        self.all_levels.remove(level)
        if len(removed_levels) > 0: self.remove_level = True
        self.lines = {key: value for key, value in self.lines.items() if key not in removed_levels}

    def update_potential_levels(self, current_index):
        self.new_potential_level = False
        timeframe = self.timeframe


        window_size = self.params[self.detection_method]['window_size'][self.timeframe2]
        if current_index < 2 * window_size + 1:
            return
        high_data = np.array(self.data['high'][self.last_resistance_index:current_index+1])
        low_data = np.array(self.data['low'][self.last_support_index:current_index+1])
        high_datalen = len(high_data)
        low_datalen = len(low_data)
        local_maxima = argrelextrema(high_data, lambda x, y: x >= y, order=window_size)[0]
        while len(local_maxima) > 0 and local_maxima[0] < window_size: local_maxima = local_maxima[1:]
        while len(local_maxima) > 0 and local_maxima[-1] >= high_datalen - window_size: local_maxima = local_maxima[:-1]
        local_minima = argrelextrema(low_data, lambda x, y: x <= y, order=window_size)[0]
        while len(local_minima) > 0 and local_minima[0] < window_size: local_minima = local_minima[1:]
        while len(local_minima) > 0 and local_minima[-1] >= low_datalen - window_size: local_minima = local_minima[:-1]
        potential_resistance_levels = []
        potential_support_levels = []
        last_index = self.last_resistance_index
        if len(local_maxima) > 0:
            potential_resistance_levels = list(high_data[local_maxima])
            self.last_resistance_index = max(local_maxima + last_index)
            for i, index in enumerate(local_maxima + last_index):
                level = potential_resistance_levels[i]
                self.new_potential_level = True
                if level in self.lines.keys():
                    self.lines[level]['cross'] = 0
                else:
                    bisect.insort(self.all_levels, level)
                    self.lines[level] = {}
                    self.lines[level]['time'] = self.data['time'][index]
                    self.lines[level]['price'] = level
                    self.lines[level]['isSupport'] = False
                    self.lines[level]['endTime'] = None
                    self.lines[level]['cross'] = 0
                    self.lines[level]['importance'] = 1
        last_index = self.last_support_index
        if len(local_minima) > 0:
            potential_support_levels = list(low_data[local_minima])
            self.last_support_index = max(local_minima + last_index)
            for i, index in enumerate(local_minima + last_index):
                self.new_potential_level = True
                level = potential_support_levels[i]
                if level in self.lines.keys():
                    self.lines[level]['importance'] += 2
                else:
                    bisect.insort(self.all_levels, level)
                    self.lines[level] = {}
                    self.lines[level]['time'] = self.data['time'][index]
                    self.lines[level]['price'] = level
                    self.lines[level]['isSupport'] = True
                    self.lines[level]['endTime'] = None
                    self.lines[level]['cross'] = 0
                    self.lines[level]['importance'] = 1
        self.potential_support_levels += potential_support_levels
        self.potential_resistance_levels += potential_resistance_levels

    def get_zones(self):
        potential_level_prices = np.array(list(self.lines.keys()))
        if self.new_potential_level:
            levels = self.aggregate_prices_to_levels(potential_level_prices, self.get_distance())
            self.recent_zones = levels
            self.concat_score()

    def concat_score(self):
        """score each zone using max level importance and update importance by max importance in zone"""
        zones = self.recent_zones
        if zones is not None:
            for i, level in enumerate(zones):
                levels = [x for x in self.lines.keys() if level['minprice'] <= x <= level['maxprice']]
                importances = [self.lines[x]['importance'] for x in levels]
                times = [self.lines[x]['time'] for x in levels]
                zone_score = np.sum(importances)
                zones[i]['importance'] = zone_score
                zones[i]['time'] = np.min(times)

    def score_lines(self, current_index):
        open_, high_, low_, close_ = [self.data[col][current_index] for col in ['open', 'high', 'low', 'close']]
        if self.score_method == 'candle':
            self.score_line_candle(open_, high_, low_, close_)
        elif self.score_method == 'swing':
            self.score_line_swing(current_index, self.window_size_score)  # 5 is good
        elif self.score_method == 'power':
            self.score_line_power(current_index, self.window_size_score)  # better be same as window_size

    def score_line_power(self, current_index, window_size):
        if not self.new_potential_level:
            return
        if self.last_resistance_index == current_index - window_size:
            high = self.data['high'][self.last_resistance_index]
            window_low = min(self.data['low'][self.last_resistance_index:current_index+1])
            distance = round((high - window_low) / self.atr, 1)
            high_minprice = high - (self.atr * self.closeness)
            high_maxprice = high + (self.atr * self.closeness)
            low_index = bisect.bisect_left(self.all_levels, high_minprice)
            high_index = bisect.bisect_right(self.all_levels, high_maxprice)
            for level in self.all_levels[low_index:high_index]:
                self.lines[level]['importance'] += distance
                self.lines[level]['importance'] = round(self.lines[level]['importance'], 1)

        if self.last_support_index == current_index - window_size:
            low = self.data['low'][self.last_support_index]
            window_high = max(self.data['high'][self.last_support_index:current_index+1])
            distance = round((window_high - low) / self.atr, 1)
            low_minprice = low - (self.atr * self.closeness)
            low_maxprice = low + (self.atr * self.closeness)
            low_index = bisect.bisect_left(self.all_levels, low_minprice)
            high_index = bisect.bisect_right(self.all_levels, low_maxprice)
            for level in self.all_levels[low_index:high_index]:
                self.lines[level]['importance'] += distance
                self.lines[level]['importance'] = round(self.lines[level]['importance'], 1)

    def score_line_candle(self, open, high, low, close):
        high_minprice = high - (self.atr * self.closeness)
        high_maxprice = high + (self.atr * self.closeness)
        low_minprice = low - (self.atr * self.closeness)
        low_maxprice = low + (self.atr * self.closeness)
        for level in self.lines.keys():
            if (level >= close and (high_minprice <= level <= high_maxprice) or
                    level <= close and (low_minprice <= level <= low_maxprice)):
                self.lines[level]["importance"] += 1

    def score_line_swing(self, current_index, window_size):
        if not self.new_potential_level:
            return
        if self.last_resistance_index == current_index - window_size:
            high = self.data['high'][self.last_resistance_index]
            high_minprice = high - (self.atr * self.closeness)
            high_maxprice = high + (self.atr * self.closeness)
            low_index = bisect.bisect_left(self.all_levels, high_minprice)
            high_index = bisect.bisect_right(self.all_levels, high_maxprice)
            for level in self.all_levels[low_index:high_index]:
                self.lines[level]['importance'] += 1

        if self.last_support_index == current_index - window_size:
            low = self.data['low'][self.last_support_index]
            low_minprice = low - (self.atr * self.closeness)
            low_maxprice = low + (self.atr * self.closeness)
            low_index = bisect.bisect_left(self.all_levels, low_minprice)
            high_index = bisect.bisect_right(self.all_levels, low_maxprice)
            for level in self.all_levels[low_index:high_index]:
                self.lines[level]['importance'] += 1

    def aggregate_prices_to_levels(self, prices, distance):
        clustering = AgglomerativeClustering(distance_threshold=distance, n_clusters=None)
        try:
            clustering.fit(prices.reshape(-1, 1))
        except ValueError:
            return None

        df = pd.DataFrame(data=prices, columns=('price',))
        df['cluster'] = clustering.labels_
        df['peak_count'] = 1

        grouped = df.groupby('cluster').agg(
            {
                'price': "min",
                'peak_count': 'sum'
            }
        ).reset_index()

        grouped2 = df.groupby('cluster').agg(
            {
                'price': "max",
                'peak_count': 'sum'
            }
        ).reset_index()
        grouped3 = df.groupby('cluster').agg(
            {
                'price': "mean",
                'peak_count': 'sum'
            }
        ).reset_index()

        grouped['meanprice'] = grouped3["price"]
        grouped['maxprice'] = grouped2["price"]
        grouped = grouped.rename(columns={"price": 'minprice'})
        return grouped.to_dict('records')


def get_pivots(data_dict, data_index):
    pivot_indices = []
    pivot_values = []
    pivot_state = -1
    new_index = 0
    if len(data_index) <= 1:
        return [], []
    for i in range(len(data_index)):
        zz_text = data_dict['zigzag_text'][i]
        if zz_text == 11:
            zz_text = 2
        if zz_text == 12:
            zz_text = 4
        if zz_text == 0:
            continue
        current_state = (zz_text - 1) // 2
        if pivot_state != current_state:
            if pivot_state == -1 and current_state >= 0:
                pivot_state = current_state
                continue
            if pivot_state == -1: pivot_state = 1 - current_state
            if new_index == 0 and pivot_state == 0:
                pivot_values.append(data_dict['high'][0])
            elif new_index == 0 and pivot_state == 1:
                pivot_values.append(data_dict['low'][0])
            else:
                pivot_values.append(data_dict['zigzag'][new_index])
            pivot_indices.append(new_index)
            pivot_state = current_state
            new_index = i
        else:
            new_index = i
    if len(pivot_indices) == 0:
        return [], []
    if pivot_indices[0] != 0:
        pivot_indices.insert(0, 0)
        if data_dict['low'][0] < pivot_values[0]:
            pivot_values.insert(0, data_dict['low'][0])
        else:
            pivot_values.insert(0, data_dict['high'][0])
    if len(data_index) - 1 not in pivot_indices and len(pivot_values) > 0:
        pivot_indices.append(len(data_index) - 1)
        if data_dict['close'][-1] > pivot_values[-1]:
            pivot_values.append(data_dict['high'][-1])
        else:
            pivot_values.append(data_dict['low'][-1])
    return pivot_indices, pivot_values


def make_new_pivots(first_pivot, interval, pivot_values, pivot_indices):
    current_pivot = first_pivot
    new_pivot_values = [pivot_values[0]]
    new_pivot_indices = [pivot_indices[0]]
    i = 0
    while i < len(pivot_indices) - 1:
        if current_pivot == 1:
            if i + 3 >= len(pivot_indices):
                new_pivot_values.append(pivot_values[i + 1])
                new_pivot_indices.append(pivot_indices[i + 1])
                i += 1
            elif pivot_values[i + 1] < pivot_values[i + 3]:
                new_pivot_values.append(pivot_values[i + 1])
                new_pivot_indices.append(pivot_indices[i + 1])
                i += 1
            elif (pivot_values[i + 2] - pivot_values[i + 1]) > (pivot_values[i] - pivot_values[i + 1]):
                new_pivot_values.append(pivot_values[i + 1])
                new_pivot_indices.append(pivot_indices[i + 1])
                i += 1
            else:
                if pivot_values[i + 2] > pivot_values[i]:
                    new_pivot_values[-1] = pivot_values[i + 2]
                    # new_pivot_indices[-1] = pivot_indices[i + 2]
                new_pivot_values.append(pivot_values[i + 3])
                new_pivot_indices.append(pivot_indices[i + 3])
                i += 3
        else:
            if i + 3 >= len(pivot_indices):
                new_pivot_values.append(pivot_values[i + 1])
                new_pivot_indices.append(pivot_indices[i + 1])
                i += 1
            elif pivot_values[i + 1] > pivot_values[i + 3]:
                new_pivot_values.append(pivot_values[i + 1])
                new_pivot_indices.append(pivot_indices[i + 1])
                i += 1
            elif (pivot_values[i + 1] - pivot_values[i + 2]) > interval * .3 and \
                    pivot_values[i + 1] - pivot_values[i + 2] > (pivot_values[i + 1] - pivot_values[i]) * .7:
                new_pivot_values.append(pivot_values[i + 1])
                new_pivot_indices.append(pivot_indices[i + 1])
                i += 1
            else:
                if pivot_values[i + 2] < pivot_values[i]:
                    new_pivot_values[-1] = pivot_values[i + 2]
                    # new_pivot_indices[-1] = pivot_indices[i + 2]
                new_pivot_values.append(pivot_values[i + 3])
                new_pivot_indices.append(pivot_indices[i + 3])
                i += 3
        current_pivot = -current_pivot
    return new_pivot_indices, new_pivot_values


def find_trend(data_dict, data_index, start_index, end_index):
    data = {x: data_dict[x][start_index:end_index] for x in ['open', 'high', 'low', 'close', 'zigzag', 'zigzag_text']}
    data['DateTime'] = data_index[start_index:end_index]
    data_len = len(data['close'])
    close_data = data['close']
    datetimes = data['DateTime']
    window_size = 5
    # datetimes = datetimes[window_size -1:]
    pivot_indices, pivot_values = get_pivots(data, datetimes)
    if len(pivot_indices) < 2:
        return 0
    sum_red_bars, sum_green_bars = 0, 0
    weighted_sum_red_bars, weighted_sum_green_bars = 0, 0
    sum_up_pct, sum_down_pct = 0, 0
    sum_up, sum_down = 0, 0
    highs = 0
    lows = 0
    higher_highs = 0
    lower_lows = 0
    last_high = 0
    last_low = 0
    if pivot_values[1] > pivot_values[0]:
        last_low = pivot_values[0]
    else:
        last_high = pivot_values[0]
    first_pivot = 0
    if len(pivot_indices) > 2:
        if pivot_values[1] > pivot_values[2]:
            first_pivot = -1
        else:
            first_pivot = 1
    elif len(pivot_indices) == 2:
        if pivot_values[1] > pivot_values[0]:
            first_pivot = -1
        else:
            first_pivot = 1
    if first_pivot == 1 and close_data[0] > pivot_values[0]:
        pivot_values[0] = close_data[0]
        pivot_indices[0] = 0
    if first_pivot == -1 and close_data[0] < pivot_values[0]:
        pivot_values[0] = close_data[0]
        pivot_indices[0] = 0
    min_price = np.min(close_data)
    max_price = np.max(close_data)
    interval = max_price - min_price
    while True:
        new_pivot_indices, new_pivot_values = make_new_pivots(first_pivot, interval, pivot_values, pivot_indices)
        if len(new_pivot_values) == len(pivot_values):
            break
        pivot_indices, pivot_values = new_pivot_indices, new_pivot_values
    high_pct = 0
    low_pct = 0
    first_price = data['open'][0]
    last_price = close_data[-1]
    overall_change = ((last_price - first_price) / first_price) * 100
    num_of_legs = len(pivot_values) - 1
    num_of_pairs = num_of_legs // 2
    if num_of_pairs <= 0:
        if overall_change > 0:
            return 3
        else:
            return -3
    high_count = 0
    high_sum = 0
    low_count = 0
    low_sum = 0
    high_len = 0
    low_len = 0
    low_legs = []
    high_legs = []
    for i in range(1, len(pivot_indices)):
        change = pivot_values[i] - pivot_values[i - 1]
        if change > 0:
            high_count += 1
            high_sum += change
            high_legs.append(change)
            high_len += pivot_indices[i] - pivot_indices[i - 1]
        else:
            low_count += 1
            low_sum += abs(change)
            low_legs.append(abs(change))
            low_len += pivot_indices[i] - pivot_indices[i - 1]
        pct_change = change / pivot_values[i - 1] * 100
        if pct_change > 0:
            if last_high == 0:
                last_high = pivot_values[i]
            else:
                if pivot_values[i] > last_high:
                    highs += 1
                    higher_highs += 1
                    high_pct += abs(pivot_values[i] - last_high) / last_high * 100
                else:
                    lows += 1
                    low_pct += abs(pivot_values[i] - last_high) / last_high * 100
                last_high = pivot_values[i]
            sum_green_bars += pivot_indices[i] - pivot_indices[i - 1]
            weighted_sum_green_bars += (pivot_indices[i] - pivot_indices[i - 1]) * pct_change
            sum_up_pct += pct_change
            sum_up += 1
        elif pct_change < 0:
            if last_low == 0:
                last_low = pivot_values[i]
            else:
                if pivot_values[i] > last_low:
                    highs += 1
                    high_pct += abs(pivot_values[i] - last_low) / last_low * 100
                else:
                    lows += 1
                    lower_lows += 1
                    low_pct += abs(pivot_values[i] - last_low) / last_low * 100
                last_low = pivot_values[i]
            sum_red_bars += pivot_indices[i] - pivot_indices[i - 1]
            weighted_sum_red_bars += (pivot_indices[i] - pivot_indices[i - 1]) * pct_change
            sum_down_pct += pct_change
            sum_down += 1
    diff_sum_pct = sum_up_pct + sum_down_pct
    # Calculating the percent changes
    if high_sum >= 4 * low_sum:
        return 3
    if low_sum >= 4 * high_sum:
        return -3
    max_increase = ((max_price - first_price) / first_price) * 100
    min_decrease = ((first_price - min_price) / first_price) * 100
    bullish_retracement = ((max_price - last_price) / max_price) * 100
    bearish_retracement = ((last_price - min_price) / min_price) * 100
    mean_price = np.mean(close_data)
    up_mean = mean_price > (np.max(close_data) + np.min(close_data)) / 2
    down_mean = mean_price < (np.max(close_data) + np.min(close_data)) / 2
    pivot_dist = [(pivot_values[i] - pivot_values[i - 1]) for i in range(1, len(pivot_values))]
    max_leg = np.max(pivot_dist)
    min_leg = np.min(pivot_dist)
    if overall_change > 0 and last_price > (
            first_price + max_price) * .4995 and first_price - min_price < last_price - first_price:
        if last_price > (first_price + max_price) * .5:
            result = 1
        else:
            result = 0
        if max_leg * .5 < abs(min_leg):
            result = -1
        if max_leg * .75 < abs(min_leg):
            return 0
        # if up_mean:
        if high_pct > low_pct:
            if highs >= 1.2 * lows or high_pct > 1.2 * low_pct:
                return (result + 2)
            else:
                return (result + 1)
        elif highs > lows:
            return (result + 1)
        if bullish_retracement < .5 * max_increase:
            return result
        return 0
    elif overall_change < 0 and last_price <= (
            first_price + min_price) * .5005 and max_price - first_price < first_price - last_price:
        result = -1
        if last_price <= (first_price + min_price) * .5:
            result = -1
        else:
            result = 0
        if abs(min_leg) * .5 < max_leg:
            result = 1
        if abs(min_leg) * .75 < max_leg:
            return 0
        # if down_mean:
        if high_pct < low_pct:
            if lows >= 1.2 * highs or 1.2 * high_pct < low_pct:
                return result - 2
            else:
                return result - 1
        elif highs < lows:
            return result - 1
        if bearish_retracement < .5 * min_decrease:
            return result
        return 0
    else:
        return 0


class FunctionCalls:
    def __init__(self):
        url = 'http://localhost:8086'
        token = 'WrSMwFo5b-ngd_gMqp1ZjGijae9QtQRKlNXd9U_8ExvcY0oVjQjZ7-dtmruJX_joU_pMzH72YUibcOX7XrvbBw=='
        org = 'TenSurf'
        self.bronze_client = InfluxClient(token, org, url, 'bronze')

    def detect_trend(self, parameters):  
        #trends range are in [-3,3]
        #-3 represents a strong downtrend, -2: moderate downtrend, -1: weak downtrend, 0: neutral, 
        #1: weak uptrend, 2: moderate uptrend, and 3: strong uptrend.
        symbol = parameters["symbol"]
        start_datetime = parameters["start_datetime"]
        end_datetime = parameters["end_datetime"]
        if end_datetime is None:
            end_datetime = datetime.now()
        if start_datetime is None:
            start_datetime =  end_datetime - timedelta(days=7)
        if isinstance(start_datetime, str):
            start_datetime = pd.to_datetime(start_datetime)
        if isinstance(end_datetime, str):
            end_datetime = pd.to_datetime(end_datetime)
        df = self.bronze_client.retrieve_db_df_between(symbol, start_datetime, end_datetime)
        if df is None or len(df) == 0:
            print('There is no data for this period of time...')
            return 0
        else:
            data_dict = df.to_dict(orient='list')
            data_index = data_dict['DateTime']
            return find_trend(data_dict, data_index, 0, len(data_index)  - 1)

    def calculate_sr(self, parameters):
        symbol = parameters["symbol"]
        timeframe = parameters["timeframe"]
        lookback_days = int(parameters["lookback_days"].split(" ")[0])
        end_datetime = datetime.now()
        start_datetime =  end_datetime - timedelta(days=lookback_days)
        df = self.bronze_client.retrieve_db_df_between(symbol, start_datetime, end_datetime)
        if df is None or len(df) == 0:
            print('There is no data for this period of time...')
            return 0
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.set_index('DateTime')
        params = {'agglomerative': {'window_size': {'1h':2,'1d':2,'1w':2,'1min':5}, 'use_maxima': True, 'merge_percent': {'1h':.5,'1d':.5,'1w':.25,'1min':.75}, 'max_cross': 2,
                                    'score': 'power','closeness': {'1h':.25,'1d':.25,'1w':.25,'1min':.25}}}
        detector = SRDetector(df, timeframe, params, 'agglomerative')
        detector.get_levels()
        levels = list(detector.lines.keys())
        start_times = [detector.lines[x]['time'] for x in levels]
        end_datetimes = [df.index[-1]]*len(levels)
        scores = [detector.lines[x]['importance'] for x in levels]
        return [levels, start_times, end_datetimes, scores]

    def stop_loss_suggestions(self, parameters):
        '''
        This function calculates stoploss based on symbol name and direction of position which is one of -1 and 1.
        '''
        symbol = parameters["symbol"]
        direction = parameters["direction"]

        end_datetime = datetime.now()
        start_datetime =  end_datetime - timedelta(days=4)
        df = self.bronze_client.retrieve_db_df_between(symbol, start_datetime, end_datetime)
        if df is None or len(df) == 0:
            print('There is no data for this period of time...')
            return {}
        df = df.iloc[-23 * 60 :] #Approximately last one day
        data_dict = df.to_dict(orient='list')
        data_index = data_dict['DateTime']
        atr = data_dict['ATR'][-1]
        stop_loss, take_profit = 0, 0
        current_index = len(data_dict['close']) - 1
        fill_price = data_dict['close'][current_index]
        neighborhood = 20
        pivot_param = .0004
        sl_lookback = 100
        ################################################################################################################################
        stoploss = {}
        if direction == 1:
            data = np.array(data_dict['low'])
            condition = -direction
            pivots = peak_valley_pivots(data, pivot_param, -pivot_param)
            if np.sum(pivots == condition):
                potential = [x for x in data[np.where(pivots == -1)[0]] if x < fill_price]
                if len(potential) > 0:
                    stoploss['pivot'] = potential[-1]
                else:
                    stoploss['pivot'] = None
            else:
                stoploss['pivot'] = None
            # swing method
            swing_indices = argrelextrema(data, lambda x, y: x < y, order=neighborhood, mode='clip')[0]
            if len(swing_indices) and swing_indices[0] < neighborhood:
                swing_indices = swing_indices[1:]
            if len(swing_indices) and swing_indices[-1] > len(data) - neighborhood // 2:
                swing_indices = swing_indices[:-1]
            for x in swing_indices[::-1]:
                if data[x] < fill_price:
                    stoploss['swing'] = data[x]
                    break
            else:
                stoploss['swing'] = None

            stoploss['min_max'] = min(data_dict['low'][- sl_lookback:])
            stoploss['atr'] = fill_price - 1.5 * atr

        elif direction == -1:
            data = np.array(data_dict['high'])
            condition = -direction
            pivots = peak_valley_pivots(data, pivot_param, -pivot_param)
            if np.sum(pivots == condition):
                potential = [x for x in data[np.where(pivots == -1)[0]] if x > fill_price]
                if len(potential) > 0:
                    stoploss['pivot'] = potential[-1]
                else:
                    stoploss['pivot'] = None
            else:
                stoploss['pivot'] = None
            # swing method
            swing_indices = argrelextrema(data, lambda x, y: x > y, order=neighborhood, mode='clip')[0]
            if len(swing_indices) and swing_indices[0] < neighborhood:
                swing_indices = swing_indices[1:]
            if len(swing_indices) and swing_indices[-1] > len(data) - neighborhood // 2:
                swing_indices = swing_indices[:-1]
            for x in swing_indices[::-1]:
                if data[x] > fill_price:
                    stoploss['swing'] = data[x]
                    break
            else:
                stoploss['swing'] = None

            stoploss['min_max'] = max(data_dict['low'][-sl_lookback:])
            stoploss['atr'] = fill_price + 1.5 * atr

        return stoploss
