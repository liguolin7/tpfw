# 配置文件路径和模型参数
import os

# 数据路径
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
RESULTS_DIR = 'results'

# 文件名
TRAFFIC_FILE = 'traffic/metr-la.csv'
WEATHER_FILE = 'weather/noaa_weather_5min.csv'

# 数据处理参数
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 模型参数
RANDOM_STATE = 42

# 模型配置
MODELS = {
    'LinearRegression': {
        'name': '线性回归'
    },
    'RandomForest': {
        'name': '随机森林',
        'params': {
            'n_estimators': 100,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
    },
    'LSTM': {
        'name': 'LSTM',
        'params': {
            'units': [50, 30],
            'batch_size': 32,
            'epochs': 1,
            'patience': 1
        }
    }
}

# 实验配置
EXPERIMENT_TYPES = ['baseline', 'enhanced']