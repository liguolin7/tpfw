# 配置文件路径和模型参数
import os

# 数据路径
RAW_DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
RESULTS_DIR = 'results'

# 文件名
TRAFFIC_FILE = 'metr-la.csv'
WEATHER_FILE = 'noaa_weather_5min.csv'

# 数据处理参数
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 模型参数
RANDOM_STATE = 42 