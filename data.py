from meteostat import Point, Daily
from datetime import datetime
import numpy as np

def data_process():
    uci = Point(33.6405, -117.8443)  # Latitude, Longitude of uci
    start = datetime(2015, 1, 1)
    end = datetime(2023, 12, 31)
    data = Daily(uci, start, end)
    data = data.fetch()

    data['snow'].fillna(0, inplace=True)
    data['wpgt'].fillna(0, inplace=True)
    data['tsun'].fillna(0, inplace=True)
    data['pres'].fillna(0, inplace=True)
    data['wdir'] = data['wdir'].interpolate(method='linear').fillna(method='bfill')

    data['day_of_year'] = data.index.dayofyear
    data['month'] = data.index.month
    data['season'] = (data['month'] % 12) // 3
    data['temp_range'] = data['tmax'] - data['tmin']
    data['is_rainy'] = (data['prcp'] > 0).astype(int)
    data['is_snowy'] = (data['snow'] > 0).astype(int)

    features = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun',
                'day_of_year', 'season', 'temp_range', 'is_rainy', 'is_snowy']
    target = ['tavg']
    dataset = data[features + target]

    return features, target, dataset

def create_dataset(data, look_back=30):
    X,y = [],[]
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), :-1])
        y.append(data[i+look_back, -1])
    return np.array(X), np.array(y)

