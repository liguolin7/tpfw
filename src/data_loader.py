import pandas as pd
import numpy as np
from config import *

def load_traffic_data():
    """加载交通数据"""
    file_path = os.path.join(RAW_DATA_DIR, TRAFFIC_FILE)
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

def load_weather_data():
    """加载天气数据"""
    file_path = os.path.join(RAW_DATA_DIR, WEATHER_FILE)
    return pd.read_csv(file_path, index_col=0, parse_dates=True) 