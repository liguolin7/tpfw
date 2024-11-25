# 配置文件路径和模型参数
import os

# 数据路径
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
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

# 性能优化参数
TRAINING_CONFIG = {
    'batch_size': 32,
    'max_epochs': 100,
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 15,
        'restore_best_weights': True
    },
    'lr_scheduler': {
        'monitor': 'val_loss',
        'factor': 0.2,
        'patience': 8,
        'min_lr': 1e-6
    },
    'validation_split': 0.2,
    'multiprocessing': True,
    'num_workers': 4
}

# 模型配置
MODEL_CONFIG = {
    'LSTM': {
        'units': [128, 64, 32],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'loss': 'mse',
        'metrics': ['mae']
    },
    'GRU': {
        'units': [128, 64, 32],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'loss': 'mse',
        'metrics': ['mae']
    },
    'Transformer': {
        'head_size': 64,
        'num_heads': 8,
        'ff_dim': 256,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'loss': 'mse',
        'metrics': ['mae']
    }
}

# 实验配置
EXPERIMENT_TYPES = ['baseline', 'enhanced']