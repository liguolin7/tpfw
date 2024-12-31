import pandas as pd
import os
import logging
from .config import RAW_DATA_DIR, TRAFFIC_FILE, WEATHER_FILE

def load_traffic_data():
    """Load traffic data"""
    try:
        file_path = os.path.join(RAW_DATA_DIR, TRAFFIC_FILE)
        traffic_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info(f"Loaded traffic data: {traffic_data.shape}")
        return traffic_data
    except FileNotFoundError:
        logging.error(f"Traffic data file not found: {file_path}")
        raise
    except pd.errors.ParserError:
        logging.error(f"Traffic data file format error: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unknown error occurred while loading traffic data: {e}")
        raise

def load_weather_data():
    """Load weather data"""
    try:
        file_path = os.path.join(RAW_DATA_DIR, WEATHER_FILE)
        weather_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info(f"Loaded weather data: {weather_data.shape}")
        return weather_data
    except FileNotFoundError:
        logging.error(f"Weather data file not found: {file_path}")
        raise
    except pd.errors.ParserError:
        logging.error(f"Weather data file format error: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unknown error occurred while loading weather data: {e}")
        raise 