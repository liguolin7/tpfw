import pandas as pd
import numpy as np
from config import *
import logging

def load_traffic_data():
    """加载METR-LA交通数据"""
    file_path = os.path.join(RAW_DATA_DIR, TRAFFIC_FILE)
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    logging.info(f"加载交通数据: {df.shape}")
    return df

def load_weather_data():
    """加载NOAA天气数据"""
    file_path = os.path.join(RAW_DATA_DIR, WEATHER_FILE)
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    logging.info(f"加载天气数据: {df.shape}")
    return df 