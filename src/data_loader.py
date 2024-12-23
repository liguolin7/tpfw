import pandas as pd
import os
import logging
from .config import RAW_DATA_DIR, TRAFFIC_FILE, WEATHER_FILE

def load_traffic_data():
    """加载交通数据"""
    try:
        file_path = os.path.join(RAW_DATA_DIR, TRAFFIC_FILE)
        traffic_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info(f"加载交通数据: {traffic_data.shape}")
        return traffic_data
    except FileNotFoundError:
        logging.error(f"交通数据文件未找到: {file_path}")
        raise
    except pd.errors.ParserError:
        logging.error(f"交通数据文件格式错误: {file_path}")
        raise
    except Exception as e:
        logging.error(f"加载交通数据时发生未知错误: {e}")
        raise

def load_weather_data():
    """加载天气数据"""
    try:
        file_path = os.path.join(RAW_DATA_DIR, WEATHER_FILE)
        weather_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info(f"加载天气数据: {weather_data.shape}")
        return weather_data
    except FileNotFoundError:
        logging.error(f"天气数据文件未找到: {file_path}")
        raise
    except pd.errors.ParserError:
        logging.error(f"天气数据文件格式错误: {file_path}")
        raise
    except Exception as e:
        logging.error(f"加载天气数据时发生未知错误: {e}")
        raise 