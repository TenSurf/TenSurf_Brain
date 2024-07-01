import pytz
import re
import pandas as pd
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import bisect
from datetime import datetime , timedelta
from sklearn.cluster import AgglomerativeClustering
from datetime import timedelta, datetime
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import argrelextrema
import numpy as np
from influxdb_client import InfluxDBClient
import requests


def get_polygon_data(ticker: str, start_datetime: pd.Timestamp = None, end_datetime: pd.Timestamp = None, timeframe: str = 'minute', api_key: str = None, data_type: str = "realtime") -> pd.DataFrame:
    start_timestamp_ms = int(start_datetime.timestamp()) * 1000
    end_timestamp_ms = int(end_datetime.timestamp()) * 1000
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{start_timestamp_ms}/{end_timestamp_ms}?apiKey={api_key}"
    response = requests.get(url)
    function_response = response.json()
    prices = [{'DateTime': datetime.fromtimestamp(result['t'] / 1000), 'open': result['o'], 'close': result['c'], 'high': result['h'], 'low': result['l'], 'volume': result['v']} for result in function_response['results']]
    return pd.DataFrame(prices)


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


def true_range(h,l,c):
    high_low = h - l
    high_close = np.abs(h-c)
    low_close = np.abs(l-c)
    return np.max([high_low, high_close, low_close])


def get_session(date):
    if date.hour >= 15 or date.hour < 6 or date.hour == 6 and date.minute < 30:
        return 'overnight'
    return 'RTH'


def is_hammer(open, high, low, close, atr):
    body_size = abs(close - open)
    candle_size = high - low
    upper_shadow = high - max(open, close)
    lower_shadow = min(open, close) - low
    if lower_shadow > atr and lower_shadow > 2 * body_size and (upper_shadow + body_size) < 0.33 * candle_size:
        return 1
    if upper_shadow > atr and upper_shadow > 2 * body_size and (lower_shadow + body_size) < 0.33 * candle_size:
        return -1
    return 0


class zigzagClass:
    def __init__(self, mode='mode1', reversal=.02, reversal_amount=10 ):
        self.all_days = []
        self.reversal = reversal
        self.reversal_amount = reversal_amount
        self.mode = mode
        self.first_low = 0
        self.first_high = 0
        self.last_high, self.last_low = 0, 0
        self.prev_low = 0
        self.prev_high = 0
        self.day7_index = 0
        self.pivot_index = 0
        self.bar_count_since_last_pivot = 0
        self.pivot_points = []
        self.recent_swing_info = {'price': 0, 'time': None, 'type': '', 'note': ''}
        self.trend = ''
        self.minimum_candles = 3

    def update_pivots(self, data_dict, data_index, current_index):
        new_leg = False
        cur_high = data_dict['high'][current_index]
        cur_low = data_dict['low'][current_index]
        day = data_index[current_index].date()
        if day not in self.all_days:
            self.all_days.append(day)
        self.bar_count_since_last_pivot += 1
        if self.first_high == 0:
            self.first_low = cur_low
            self.first_high = cur_high
            self.last_high, self.last_low = self.first_high, self.first_low
            return
        if self.trend == '':
            if cur_low > self.first_high:
                self.pivot_points = [{'session':get_session(data_index[current_index]), 'date':data_index[current_index], 'index':current_index, 'value': self.last_high, 'trend': 'down'}]
                self.trend = 'up'
                self.last_high = cur_high
                self.pivot_index = current_index
                self.bar_count_since_last_pivot = 0
                self.low_of_current_up_trend = cur_low
                return
            elif cur_high < self.first_low:
                self.pivot_points = [{'session':get_session(data_index[current_index]), 'date':data_index[current_index], 'index':current_index, 'value': self.last_low, 'trend': 'up'}]
                self.trend = 'down'
                self.last_low = cur_low
                self.pivot_index = current_index
                self.bar_count_since_last_pivot = 0
                self.high_of_current_down_trend = cur_high
                return
        if self.trend == 'up':
            reversal_price = self.last_high - (self.last_high * self.reversal / 100) if self.mode == 'mode1' else self.last_high - self.reversal_amount
            if self.mode == 'mode2':
                if cur_high >= self.last_high:
                    self.last_high = cur_high
                    self.pivot_index = current_index
                elif cur_high < self.last_high:
                    if self.last_high - cur_low > self.reversal_amount and self.bar_count_since_last_pivot >= 2:
                        self.pivot_points.append({'session':get_session(data_index[self.pivot_index]), 'date':data_index[self.pivot_index],'index':self.pivot_index, 'value': self.last_high, 'trend': self.trend})
                        new_leg = True
                        self.trend = 'down'
                        self.last_low = cur_low
                        self.pivot_index = current_index
                        self.bar_count_since_last_pivot = 0
                        self.high_of_current_down_trend = cur_high
                elif cur_low < self.low_of_current_up_trend:
                    self.pivot_points.append({'session':get_session(data_index[self.pivot_index]), 'date':data_index[self.pivot_index],'index':self.pivot_index, 'value': self.last_high, 'trend': self.trend})
                    new_leg = True
                    self.trend = 'down'
                    self.last_low = cur_low
                    self.pivot_index = current_index
                    self.bar_count_since_last_pivot = 0
                    self.high_of_current_down_trend = cur_high
                self.low_of_current_up_trend = min(self.low_of_current_up_trend, cur_low)
            else:
                if cur_high >= self.last_high:
                    self.last_high = cur_high
                    self.pivot_index = current_index
                    self.bar_count_since_last_pivot = 0
                ############################################################################################ Find Peak
                if cur_low < reversal_price and self.bar_count_since_last_pivot >= self.minimum_candles:
                    self.pivot_points.append({'session':get_session(data_index[self.pivot_index]), 'date':data_index[self.pivot_index],'index':self.pivot_index, 'value': self.last_high, 'trend': self.trend})
                    new_leg = True
                    self.trend = 'down'
                    ind = self.minimum_candles - 1 - np.argmin(
                        data_dict['low'][current_index - self.minimum_candles + 1: current_index + 1][::-1])
                    self.pivot_index = current_index - self.minimum_candles + 1 + ind
                    self.last_low = data_dict['low'][self.pivot_index]
                    self.bar_count_since_last_pivot = 0
        elif self.trend == 'down':
            reversal_price = self.last_low + (self.last_low * self.reversal / 100) if self.mode == 'mode1' else self.last_low + self.reversal_amount
            if self.mode == 'mode2':
                if cur_low <= self.last_low:
                    self.last_low = cur_low
                    self.pivot_index = current_index
                elif cur_low > self.last_low:
                    if cur_high - self.last_low > self.reversal_amount and self.bar_count_since_last_pivot >= 2:
                        self.pivot_points.append({'session':get_session(data_index[self.pivot_index]), 'date':data_index[self.pivot_index],'index':self.pivot_index, 'value': self.last_low, 'trend': self.trend})
                        new_leg = True
                        self.trend = 'up'
                        self.last_high = cur_high
                        self.pivot_index = current_index
                        self.bar_count_since_last_pivot = 0
                        self.low_of_current_up_trend = cur_low
                elif cur_high > self.high_of_current_down_trend:
                    self.pivot_points.append({'session':get_session(data_index[self.pivot_index]), 'date':data_index[self.pivot_index],'index':self.pivot_index, 'value': self.last_low, 'trend': self.trend})
                    new_leg = True
                    self.trend = 'up'
                    self.last_high = cur_high
                    self.pivot_index = current_index
                    self.bar_count_since_last_pivot = 0
                    self.low_of_current_up_trend = cur_low
                self.high_of_current_down_trend = max(self.high_of_current_down_trend, cur_high)
            else:
                if cur_low <= self.last_low:
                    self.last_low = cur_low
                    self.pivot_index = current_index
                    self.bar_count_since_last_pivot = 0
                ############################################################################################ Find Valley
                if cur_high > reversal_price and self.bar_count_since_last_pivot >= self.minimum_candles:
                    self.pivot_points.append({'session':get_session(data_index[self.pivot_index]), 'date':data_index[self.pivot_index],'index':self.pivot_index, 'value': self.last_low, 'trend': self.trend})
                    new_leg = True
                    self.trend = 'up'
                    ind = self.minimum_candles - 1 - np.argmax(
                        data_dict['high'][current_index - self.minimum_candles + 1: current_index + 1][::-1])
                    self.pivot_index = current_index - self.minimum_candles + 1 + ind
                    self.last_high = data_dict['high'][self.pivot_index]
                    self.bar_count_since_last_pivot = 0
        if new_leg:
            swing_type = ''
            p_value = self.pivot_points[-1]['value']
            if len(self.pivot_points) >= 3:
                if self.pivot_points[-1]['trend'] == 'up':
                    if p_value >= self.pivot_points[-3]['value']:
                        swing_type = 'HH'
                    else:
                        swing_type = 'LH'
                elif p_value >= self.pivot_points[-3]['value']:
                    swing_type = 'HL'
                else:
                    swing_type = 'LL'
            self.recent_swing_info = {'price': p_value,
                                          'time': self.pivot_points[-1]['date'], 'type': swing_type, 'note': ''}
        return new_leg

    def get_mean_leg(self, current_date):
        if len(self.pivot_points) == 0:
            return 0
        while self.pivot_points[-1]['date'] - self.pivot_points[self.day7_index]['date'] > timedelta(days=7):
            self.day7_index += 1
        current_session = get_session(current_date)
        legs = [abs(self.pivot_points[i]['value'] - self.pivot_points[i + 1]['value']) for i in range(self.day7_index, len(self.pivot_points) - 1) if self.pivot_points[i+1]['session'] == current_session]
        mean_leg = np.mean(legs)
        # print(self.pivot_dates, mean_leg, '-----------------------')
        return mean_leg


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
        self.current_date = None
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
            self.current_date = self.data['time'][i]
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
                    self.lines[level]['detect_time'] = self.current_date
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
                    self.lines[level]['detect_time'] = self.current_date
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
    zigzag = zigzagClass(mode='mode1', reversal=.02)
    for current_index in range(len(data_dict['close'])):
        zigzag.update_pivots(data_dict, data_index,current_index)
    return [zigzag.pivot_points[x]['index'] for x in range(len(zigzag.pivot_points))], [zigzag.pivot_points[x]['value'] for x in range(len(zigzag.pivot_points))]


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
    # data = {x: data_dict[x][start_index:end_index] for x in ['open', 'high', 'low', 'close', 'zigzag', 'zigzag_text']}
    data = {x: data_dict[x][start_index:end_index] for x in ['open', 'high', 'low', 'close']}
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


config = {
	"ES": {
        "point_value": 50,
        "tick_size": 0.25
    },
    "NQ": {
        "point_value": 20,
        "tick_size": 0.25
    },
    "RT": {
        "point_value": 50,
        "tick_size": 0.1
    },
    "YM": {
        "point_value": 5,
        "tick_size": 1
    },
	"MES": {
        "point_value": 5,
        "tick_size": 0.25
    },
    "MNQ": {
        "point_value": 2,
        "tick_size": 0.25
    },
    "CL": {
        "point_value": 1000,
        "tick_size": 0.01
    },
    "MCL": {
        "point_value": 100,
        "tick_size": 0.01
    },
    "GC": {
        "point_value": 1000,
        "tick_size": 0.01
    },
    "MGC": {
        "point_value": 10,
        "tick_size": 0.1
    },
    "NG": {
        "point_value": 10000,
        "tick_size": 0.001
    },
    "M2K": {
        "point_value": 5,
        "tick_size": 0.1
    },
    "ZB": {
        "point_value": 31.25,
        "tick_size": 0.01
    },
    "ZC": {
        "point_value": 25,
        "tick_size": 0.25
    },
    "ZM": {
        "point_value": 100,
        "tick_size": 0.1
    },
    "ZL": {
        "point_value": 600,
        "tick_size": 0.01
    },
    "ZS": {
        "point_value": 50,
        "tick_size": 0.25
    },
    "ZW": {
        "point_value": 50,
        "tick_size": 0.25
    },
    "HG": {
        "point_value": 25000,
        "tick_size": 0.005
    },
    "MHG": {
        "point_value": 2500,
        "tick_size": 0.005
    },
    "HE": {
        "point_value": 400,
        "tick_size": 0.025
    },
    "LE": {
        "point_value": 400,
        "tick_size": 0.025
    },
    "SI": {
        "point_value": 5000,
        "tick_size": 0.005
    },
    "SIL": {
        "point_value": 1000,
        "tick_size": 0.005
    },
    "MYM": {
        "point_value": 0.5,
        "tick_size": 1
    },
    "6A": {
        "point_value": 125000,
        "tick_size": 5e-05
    },
    "6B": {
        "point_value": 62500,
        "tick_size": 0.0001
    },
    "6C": {
        "point_value": 100000,
        "tick_size": 5e-05
    },
    "6E": {
        "point_value": 125000,
        "tick_size": 5e-05
    },
    "6J": {
        "point_value": 125000,
        "tick_size": 5e-05
    }
}


class FunctionCalls:
    def __init__(self):
        url = 'http://73.241.173.17:8086'
        token = 'WrSMwFo5b-ngd_gMqp1ZjGijae9QtQRKlNXd9U_8ExvcY0oVjQjZ7-dtmruJX_joU_pMzH72YUibcOX7XrvbBw=='
        org = 'TenSurf'
        self.bronze_client = InfluxClient(token, org, url, 'bronze')
        self.url = url
        self.token = token
        self.org = org

    def get_data(self, symbol, start_datetime, end_datetime):
        if symbol in ['ES', 'NQ', 'GC', 'CL', 'YM', 'RTY']:
            df = self.bronze_client.retrieve_db_df_between(symbol, start_datetime, end_datetime)
        else:
            polygon_api_key = "6lpCMsrDOzmm6PPpSkci73RfUvEeU9y_"
            df = get_polygon_data(ticker='AAPL', start_datetime=start_datetime, end_datetime=end_datetime, timeframe='minute',
                                                     data_type='historical', api_key=polygon_api_key)
        return df
    
    def detect_trend(self, parameters):
        # trends range are in [-3,3]
        # -3 represents a strong downtrend, -2: moderate downtrend, -1: weak downtrend, 0: neutral,
        # 1: weak uptrend, 2: moderate uptrend, and 3: strong uptrend.
        symbol = parameters["symbol"]
        start_datetime = parameters["start_datetime"]
        end_datetime = parameters["end_datetime"]
        if end_datetime is None:
            end_datetime = datetime.now()
        if start_datetime is None:
            start_datetime = end_datetime - timedelta(days=7)
        if isinstance(start_datetime, str):
            start_datetime = pd.to_datetime(start_datetime)
        if isinstance(end_datetime, str):
            end_datetime = pd.to_datetime(end_datetime)
        df = self.get_data(symbol, start_datetime, end_datetime)
        if df is None or len(df) == 0:
            print('There is no data for this period of time...')
            return 0
        else:
            data_dict = df.to_dict(orient='list')
            data_index = data_dict['DateTime']
            return find_trend(data_dict, data_index, 0, len(data_index) - 1)

    # TODO: checking the output values
    def calculate_sr(self, parameters):
        symbol = parameters["symbol"]
        timeframe = parameters["timeframe"]
        lookback_days = int(parameters["lookback_days"].split(" ")[0])
        end_datetime = datetime.now()
        start_datetime = end_datetime - timedelta(days=lookback_days)
        df = self.get_data(symbol, start_datetime, end_datetime)
        if df is None or len(df) == 0:
            print('There is no data for this period of time...')
            return 0
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.set_index('DateTime')
        params = {'agglomerative': {'window_size': {'1h': 2, '1d': 2, '1w': 2, '1min': 5}, 'use_maxima': True,
                                    'merge_percent': {'1h': .5, '1d': .5, '1w': .25, '1min': .75}, 'max_cross': 2,
                                    'score': 'power', 'closeness': {'1h': .25, '1d': .25, '1w': .25, '1min': .25}}}
        detector = SRDetector(df, timeframe, params, 'agglomerative')
        detector.get_levels()
        levels = list(detector.lines.keys())
        start_times = [detector.lines[x]['time'] for x in levels]
        detect_times = [detector.lines[x]['detect_time'] for x in levels]
        end_datetimes = [df.index[-1]] * len(levels)
        scores = [detector.lines[x]['importance'] for x in levels]
        return [levels, start_times, detect_times, end_datetimes, scores]

    def round(self, x):
        coef = 1 / self.ticksize
        return round(x * coef) / coef
    
    def calculate_sl(self, parameters):
        def get_vwap_stop(vwap_level_names, vwap_values, fill_price, direction):
            delta_vwap = vwap_values[1] - vwap_values[0]
            if direction == 1:
                for i in range(1, 8):
                    if vwap_values[i] <= fill_price <= vwap_values[i + 1]:
                        return vwap_values[i - 1], vwap_level_names[i - 1]
                if fill_price > vwap_values[-1]:
                    dist = np.floor((fill_price - vwap_values[-1]) / delta_vwap)
                    name = f'{vwap_level_names[-1][:-1]}{int(4 + dist - 1)}'
                    return vwap_values[-1] + (dist - 1) * delta_vwap, name
                if fill_price > vwap_values[0]:
                    return vwap_values[0] - delta_vwap, f'{vwap_level_names[0][:-1]}5'
                dist = np.ceil((vwap_values[0] - fill_price) / delta_vwap)
                name = f'{vwap_level_names[0][:-1]}{int(4 + dist + 1)}'
                return vwap_values[0] - (dist + 1) * delta_vwap, name
            else:  # direction == -1
                for i in range(1, 8):
                    if vwap_values[i - 1] <= fill_price <= vwap_values[i]:
                        return vwap_values[i + 1], vwap_level_names[i + 1]
                if fill_price < vwap_values[0]:
                    dist = np.floor((vwap_values[0] - fill_price) / delta_vwap)
                    name = f'{vwap_level_names[0][:-1]}{int(4 + dist - 1)}'
                    return vwap_values[0] - (dist - 1) * delta_vwap, name
                if fill_price < vwap_values[-1]:
                    return vwap_values[-1] + delta_vwap, f'{vwap_level_names[-1][:-1]}5'
                dist = np.ceil((fill_price - vwap_values[-1]) / delta_vwap)
                name = f'{vwap_level_names[-1][:-1]}{int(4 + dist + 1)}'
                return vwap_values[-1] + (dist + 1) * delta_vwap, name

        symbol = parameters["symbol"]
        ticksize = config[symbol]['tick_size']
        self.ticksize = ticksize
        direction = parameters["direction"]
        method = parameters["method"] if 'method' in parameters else 'all'
        if method is None or method == '':
            method = 'all'
        neighborhood = parameters["neighborhood"] if 'neighborhood' in parameters else 20
        atr_coef = parameters["atr_coef"] if 'atr_coef' in parameters else 1.5
        lookback = parameters["lookback"] if 'lookback' in parameters else 100
        min_sl_ticks = parameters["min_sl_ticks"] if 'min_sl_ticks' in parameters else 4
        minimum_risk = min_sl_ticks * ticksize
        end_datetime = datetime.now()
        start_datetime = end_datetime - timedelta(days=4)
        df = self.bronze_client.retrieve_db_df_between(symbol, start_datetime, end_datetime)
        if df is None or len(df) == 0:
            print('There is no data for this period of time...')
            return {}
        siver_client = InfluxClient(self.token, self.org, self.url, 'silver')
        df = df.iloc[-23 * 60:]  # Approximately last one day
        data_dict = df.to_dict(orient='list')
        data_index = data_dict['DateTime']
        atr = data_dict['ATR'][-1]
        current_index = len(data_dict['close']) - 1
        fill_price = data_dict['close'][current_index]
        answer = {'sl': [], 'risk': [], 'info': []}
        sl_dict = {}
        if method in ['swing', 'all']:
            if direction == 1:
                data = np.array(data_dict['low'])
                swing_indices = argrelextrema(data, lambda x, y: x < y, order=neighborhood, mode='clip')[0]
                if len(swing_indices) and swing_indices[0] < neighborhood:
                    swing_indices = swing_indices[1:]
                if len(swing_indices) and swing_indices[-1] > len(data) - neighborhood // 2:
                    swing_indices = swing_indices[:-1]
                for x in swing_indices[::-1]:
                    if data[x] < fill_price - minimum_risk:
                        sl_dict['swing'] = data[x]
                        answer['sl'].append(self.round(data[x]))
                        answer['risk'].append(self.round(abs(fill_price - data[x])))
                        answer['info'].append(
                            f'calculated based on low swing with neighborhood parameter of {neighborhood} candles')
            else:
                data = np.array(data_dict['high'])
                swing_indices = argrelextrema(data, lambda x, y: x > y, order=neighborhood, mode='clip')[0]
                if len(swing_indices) and swing_indices[0] < neighborhood:
                    swing_indices = swing_indices[1:]
                if len(swing_indices) and swing_indices[-1] > len(data) - neighborhood // 2:
                    swing_indices = swing_indices[:-1]
                for x in swing_indices[::-1]:
                    if data[x] > fill_price + minimum_risk:
                        sl_dict['swing'] = data[x]
                        answer['sl'].append(self.round(data[x]))
                        answer['risk'].append(self.round(abs(fill_price - data[x])))
                        answer['info'].append(
                            f'calculated based on high swing with neighborhood parameter of {neighborhood} candles')
        if method in ['minmax', 'all']:
            if direction == 1:
                mm_stop = min(data_dict['low'][- lookback:])
                sl_dict['minmax'] = mm_stop
                answer['sl'].append(self.round(mm_stop))
                answer['risk'].append(self.round(abs(fill_price - mm_stop)))
                answer['info'].append(f'calculated based on minimum low price of previous {lookback} candles')

            else:
                mm_stop = min(data_dict['high'][- lookback:])
                sl_dict['minmax'] = mm_stop
                answer['sl'].append(self.round(mm_stop))
                answer['risk'].append(self.round(abs(fill_price - mm_stop)))
                answer['info'].append(f'calculated based on maximum high price of previous {lookback} candles')

        if method in ['atr', 'all']:
            if direction == 1:
                atr_stop = fill_price - atr_coef * atr
            else:
                atr_stop = fill_price + atr_coef * atr
            sl_dict['atr'] = atr_stop
            answer['sl'].append(self.round(atr_stop))
            answer['risk'].append(self.round(abs(fill_price - atr_stop)))
            answer['info'].append(f'calculated based on ATR with length 14 multiplied by the coefficient {atr_coef}')
        if method in ['DVWAP_band', 'all']:
            vwap_level_names = [f'VWAP_Bottom_Band_{i}' for i in range(4, 0, -1)] + ['VWAP'] + [f'VWAP_Top_Band_{i}'
                                                                                                    for
                                                                                                    i in
                                                                                                    range(1, 5)]
            vwap_values = [data_dict[x][-1] for x in vwap_level_names]
            vwap_stop, level_name = get_vwap_stop(vwap_level_names, vwap_values, fill_price, direction)
            answer['sl'].append(self.round(vwap_stop))
            answer['risk'].append(self.round(abs(fill_price - vwap_stop)))
            if level_name == 'VWAP':
                level_number = 0
            elif 'Top' in level_name:
                level_number = int(level_name.split('_')[-1])
            else:
                level_number = -1 * int(level_name.split('_')[-1])
            if direction == 1:
                if level_number >= 0:
                    cur_level = level_number + 2
                    bottop = level_number + 1
                else:
                    cur_level = level_number + 1
                    bottop = level_number
                    if level_number == -1: cur_level += 1
            else:
                if level_number <= 0:
                    cur_level = level_number - 2
                    bottop = level_number - 1
                else:
                    cur_level = level_number - 1
                    bottop = level_number
                    if level_number == 1: cur_level -= 1

            if direction == 1:
                answer['info'].append(
                    f'calculated based on {level_name} as the bottom of vwap zone {bottop} (current price is inside vwap zone {cur_level})')
            else:
                answer['info'].append(
                    f'calculated based on {level_name} as the ceiling of vwap zone {bottop} (current price is inside vwap zone {cur_level})')
        if method in ['WVWAP_band', 'all']:
            vwap_level_names = [f'WVWAP_Bottom_Band_{i}' for i in range(4, 0, -1)] + ['WVWAP'] + [
                f'WVWAP_Top_Band_{i}'
                for i in
                range(1, 5)]
            vwap_values = [data_dict[x][-1] for x in vwap_level_names]
            vwap_stop, level_name = get_vwap_stop(vwap_level_names, vwap_values, fill_price, direction)
            answer['sl'].append(self.round(vwap_stop))
            answer['risk'].append(self.round(abs(fill_price - vwap_stop)))
            if level_name == 'WVWAP':
                level_number = 0
            elif 'Top' in level_name:
                level_number = int(level_name.split('_')[-1])
            else:
                level_number = -1 * int(level_name.split('_')[-1])
            if direction == 1:
                if level_number >= 0:
                    cur_level = level_number + 2
                    bottop = level_number + 1
                else:
                    cur_level = level_number + 1
                    bottop = level_number
                    if level_number == -1: cur_level += 1
            else:
                if level_number <= 0:
                    cur_level = level_number - 2
                    bottop = level_number - 1
                else:
                    cur_level = level_number - 1
                    bottop = level_number
                    if level_number == 1: cur_level -= 1

            if direction == 1:
                answer['info'].append(
                    f'calculated based on {level_name} as the bottom of wvwap zone {bottop} (current price is inside wvwap zone {cur_level})')
            else:
                answer['info'].append(
                    f'calculated based on {level_name} as the ceiling of wvwap zone {bottop} (current price is inside wvwap zone {cur_level})')
        if method in ['zigzag', 'level', 'all']:
            # Read Silver data
            sdf = siver_client.retrieve_db_df_between(symbol, start_datetime, end_datetime)
            sdf = sdf.iloc[-1]
        if method in ['zigzag', 'all']:
            if direction == 1:
                zz_stop = fill_price  - sdf['ZZ_rth_last5']
            else:
                zz_stop = fill_price + sdf['ZZ_rth_last5']
            answer['sl'].append(self.round(zz_stop))
            answer['risk'].append(self.round(abs(fill_price - zz_stop)))
            answer['info'].append(f'Calculated based on the last 5 zigzag legs of the same day of the week')
            if direction == 1:
                zz_stop = fill_price  - sdf['ZZ_rth_daily']
            else:
                zz_stop = fill_price + sdf['ZZ_rth_daily']
            answer['sl'].append(self.round(zz_stop))
            answer['risk'].append(self.round(abs(fill_price - zz_stop)))
            answer['info'].append(f'Calculated based on the last session zigzag')

        if method in ['level', 'all']:
            vwap_level_names = [f'VWAP_Bottom_Band_{i}' for i in range(4, 0, -1)] + ['VWAP'] + [f'VWAP_Top_Band_{i}' for
                                                                                                i in
                                                                                                range(1, 5)]
            vwap_level_names += [f'WVWAP_Bottom_Band_{i}' for i in range(4, 0, -1)] + ['WVWAP'] + [f'WVWAP_Top_Band_{i}'
                                                                                                   for i in
                                                                                                   range(1, 5)]
            vwap_values = [data_dict[x][-1] for x in vwap_level_names]
            initial_level_names = ['VP_POC', 'VP_VAL', 'VP_VAH', 'Overnight_high', 'Overnight_low', 'Overnight_mid',
                                   'initial_balance_high',
                                   'initial_balance_low', 'initial_balance_mid', 'prev_session_max', 'prev_session_min',
                                   'prev_session_mid']
            level_names = []
            level_values = []
            for i in range(len(initial_level_names)):
                level = initial_level_names[i]
                if direction == 1 and sdf[level] < fill_price - minimum_risk or direction == -1 and sdf[
                    level] > fill_price + minimum_risk:
                    level_values.append(sdf[level])
                    level_names.append(level)
            for i in range(len(vwap_level_names)):
                level = vwap_level_names[i]
                if direction == 1 and vwap_values[i] < fill_price - minimum_risk or direction == -1 and vwap_values[
                    i] > fill_price + minimum_risk:
                    level_values.append(vwap_values[i])
                    level_names.append(level)
            for x, y in zip(level_names, level_values):
                answer['sl'].append(self.round(y))
                answer['risk'].append(self.round(abs(fill_price - y)))
                answer['info'].append(f'calculated based on the level {x}')

            best_level = {}
            for col in [ 'weekly_SR', 'daily_SR', 'hourly_SR','5min_SR']:
                if col not in sdf:
                    continue
                sr_levels = eval(sdf[col])
                for i, val in enumerate(sr_levels['values']):
                    if direction == 1 and val < fill_price - minimum_risk:
                        if col not in best_level or val > best_level[col]['sl']:
                            best_level[col] = {'sl': val, 'info':f'calculate based on {" ".join(col.split("_"))} level starting from {sr_levels["start_time"][i]}'}
                    if direction == -1 and val > fill_price + minimum_risk:
                        if col not in best_level or val < best_level[col]['sl']:
                            best_level[col] = {'sl': val, 'info':f'calculate based on {" ".join(col.split("_"))} level starting from {sr_levels["start_time"][i]}'}
            for col in best_level:
                answer['sl'].append(self.round(best_level[col]['sl']))
                answer['risk'].append(self.round(abs(fill_price - best_level[col]['sl'])))
                answer['info'].append(best_level[col]['info'])
        if direction == 1:
            sorted_items = sorted(zip(answer['sl'], answer['risk'], answer['info']), key=lambda x: x[0], reverse=True)
        else:
            sorted_items = sorted(zip(answer['sl'], answer['risk'], answer['info']), key=lambda x: x[0])
        return {'sl': [item[0] for item in sorted_items],
                'risk': [item[1] for item in sorted_items],
                'info': [item[2] for item in sorted_items]}

    def calculate_tp(self, parameters):
        symbol = parameters["symbol"]
        ticksize = config[symbol]['tick_size']
        self.ticksize = ticksize
        direction = parameters["direction"]
        sl = parameters["stoploss"]
        method = parameters["method"] if 'method' in parameters else 'all'
        end_datetime = datetime.now()
        start_datetime = end_datetime - timedelta(days=4)
        df = self.bronze_client.retrieve_db_df_between(symbol, start_datetime, end_datetime)
        if df is None or len(df) == 0:
            print('There is no data for this period of time...')
            return {}
        siver_client = InfluxClient(self.token, self.org, self.url, 'silver')
        df = df.iloc[-23 * 60:]  # Approximately last one day
        data_dict = df.to_dict(orient='list')
        data_index = data_dict['DateTime']
        atr = data_dict['ATR'][-1]
        current_index = len(data_dict['close']) - 1
        fill_price = data_dict['close'][current_index]
        risk = abs(fill_price - sl)
        answer = {'tp': [], 'info': []}
        sdf = siver_client.retrieve_db_df_between(symbol, start_datetime, end_datetime)
        sdf = sdf.iloc[-1]
        answer = {'tp': [], 'info': []}
        vwap_level_names = [f'VWAP_Bottom_Band_{i}' for i in range(4, 0, -1)] + ['VWAP'] + [f'VWAP_Top_Band_{i}' for
                                                                                            i in
                                                                                            range(1, 5)]
        vwap_level_names += [f'WVWAP_Bottom_Band_{i}' for i in range(4, 0, -1)] + ['WVWAP'] + [f'WVWAP_Top_Band_{i}'
                                                                                               for i in
                                                                                               range(1, 5)]
        vwap_values = [data_dict[x][-1] for x in vwap_level_names]
        initial_level_names = ['VP_POC', 'VP_VAL', 'VP_VAH', 'Overnight_high', 'Overnight_low', 'Overnight_mid',
                               'initial_balance_high',
                               'initial_balance_low', 'initial_balance_mid', 'prev_session_max', 'prev_session_min',
                               'prev_session_mid']
        level_names = []
        level_values = []
        for i in range(len(initial_level_names)):
            level = initial_level_names[i]
            if direction == 1 and sdf[level] > fill_price + risk or direction == -1 and sdf[
                level] < fill_price - risk:
                level_values.append(sdf[level])
                level_names.append(level)
        for i in range(len(vwap_level_names)):
            level = vwap_level_names[i]
            if direction == 1 and  vwap_values[i] > fill_price + risk or direction == -1 and vwap_values[i] < fill_price - risk:
                level_values.append(vwap_values[i])
                level_names.append(level)
        for x, y in zip(level_names, level_values):
            answer['tp'].append(self.round(y))
            answer['info'].append(f'calculated based on the level {x}')

        best_level = {}
        for col in [ 'weekly_SR', 'daily_SR', 'hourly_SR','5min_SR']:
            if col not in sdf:
                continue
            sr_levels = eval(sdf[col])
            for i, val in enumerate(sr_levels['values']):
                if direction == 1 and val > fill_price + risk:
                    if col not in best_level or val < best_level[col]['tp']:
                        best_level[col] = {'tp': val, 'info':f'calculate based on {" ".join(col.split("_"))} level starting from {str(sr_levels["start_time"][i])[:19]}'}
                if direction == -1 and val > fill_price + risk:
                    if col not in best_level or val < best_level[col]['tp']:
                        best_level[col] = {'tp': val, 'info':f'calculate based on {" ".join(col.split("_"))} level starting from {str(sr_levels["start_time"][i])[:19]}'}
        for col in best_level:
            answer['tp'].append(self.round(best_level[col]['tp']))
            answer['info'].append(best_level[col]['info'])
        if direction == -1:
            sorted_items = sorted(zip(answer['tp'], answer['info']), key=lambda x: x[0], reverse=True)
        else:
            sorted_items = sorted(zip(answer['tp'], answer['info']), key=lambda x: x[0])
        return {'tp': [item[0] for item in sorted_items],
                'info': [item[1] for item in sorted_items]}

    def get_bias(self, parameters):
        def convert_trend(x):
            if abs(x) < .2:
                return 0
            return int( np.sign(x))
#             if abs(x)  .4:
#                 return 1 * np.sign(x)
#             if abs(x) < .6:
#                 return 2 * np.sign(x)
#             return 3 * np.sign(x)

        symbol = parameters["symbol"]
        ticksize = config[symbol]['tick_size']
        self.ticksize = ticksize
        method = parameters["method"] if 'method' in parameters else 'all'
        end_datetime = datetime.now()
        start_datetime = end_datetime - timedelta(days=4)
        df = self.bronze_client.retrieve_db_df_between(symbol, start_datetime, end_datetime)
        if df is None or len(df) == 0:
            print('There is no data for this period of time...')
            return {}
        siver_client = InfluxClient(self.token, self.org, self.url, 'silver')
        df = df.iloc[-46 * 60:]  # Approximately last one day
        data_dict = df.to_dict(orient='list')
        data_index = data_dict['DateTime']
        atr = data_dict['ATR'][-1]
        current_index = len(data_dict['close']) - 1
        cur_close = data_dict['close'][current_index]
        answer = {'bias': [], 'info': []}
        sdf = siver_client.retrieve_db_df_between(symbol, start_datetime, end_datetime)
        if method in ['weekly VWAP', 'all']:
            bias = 0
            if cur_close > data_dict['VWAP_Top_Band_2'][current_index]:
                bias = 1
            elif cur_close < data_dict['VWAP_Bottom_Band_2'][current_index]:
                bias = -1
            else:
                for current_index in range(len(data_dict['close']) - 1, -1, -1):
                    cur_high = data_dict['high'][current_index]
                    cur_low = data_dict['low'][current_index]
                    if cur_high >= data_dict['VWAP_Top_Band_2'][current_index]:
                        bias = -1
                        break
                    if cur_low <= data_dict['VWAP_Bottom_Band_2'][current_index]:
                        bias = 1
                        break
            info = 'Calculated based on weekly VWAP'
            answer['bias'].append(bias)
            answer['info'].append(info)
        if method in ['volume profile', 'all']:
            poc = sdf['VP_POC'].iloc[current_index]
            val = sdf['VP_VAL'].iloc[current_index]
            vah = sdf['VP_VAH'].iloc[current_index]
            bias = 0
            if cur_close > vah:
                bias = 1
            elif cur_close < val:
                bias = -1
            else:
                for current_index in range(len(data_dict['close']) - 1, -1, -1):
                    cur_high = data_dict['high'][current_index]
                    cur_low = data_dict['low'][current_index]
                    if cur_high >= vah:
                        bias = -1
                        break
                    if cur_low <= val:
                        bias = 1
                        break
            info = 'Calculated based on VAL/VAH of volume profile'
            answer['bias'].append(bias)
            answer['info'].append(info)
        if method in ['trend', 'all']:
            short_trend = convert_trend(sdf.iloc[-1]['short_term_trend_prob'])
            mid_trend = convert_trend(sdf.iloc[-1]['mid_term_trend_prob'])
            long_trend = convert_trend(sdf.iloc[-1]['long_term_trend_prob'])
            answer['bias'] += [short_trend, mid_trend, long_trend]
            answer['info'] += [f'Calculated based on {x} trend' for x in ['short term', 'mid term', 'long term']]
        if method in ['zigzag', 'all']:
            columns = ['trend_last_hour']
            for col in columns:
                trend = sdf.iloc[-1][col]
                answer['bias'].append(1 if trend > .2 else -1 if trend < -.2 else 0)
                answer['info'].append(f'Calculated based on trend of {" ".join(x for x in col[6:].split("_"))}')
        if method in ['counter', 'all']:
            bid_number = np.sum(df['bid_number'].iloc[-20:])
            ask_number = np.sum(df['ask_number'].iloc[-20:])
            bid_volume = np.sum(df['bid_volume'].iloc[-20:])
            ask_volume = np.sum(df['ask_volume'].iloc[-20:])
            # main calculations
            bid_ratio = bid_volume / bid_number
            ask_ratio = ask_volume / ask_number
            power_ratio = bid_ratio / ask_ratio if bid_ratio > ask_ratio else -ask_ratio / bid_ratio
            counter_ratio = bid_volume / ask_volume if bid_volume > ask_volume else - ask_volume / bid_volume
            trend = 0
            if power_ratio > 1.02 and counter_ratio > 1.02:
                trend = 1
            if power_ratio < -1.02 and counter_ratio < -1.02:
                trend = -1
            answer['bias'].append(trend)
            answer['info'].append(f'Calculated based on counter ratio and power ratio')
        if method in ['micro-composite', 'all']:
            pass
            # microcomposite = MicroComposite(self.config)
        answer['combined_bias'] = 1 if np.sum(answer['bias']) > 0 else -1 if np.sum(answer['bias']) < 0 else 0
        return answer
