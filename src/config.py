# 配置文件路径和模型参数
import os
import tensorflow as tf
import tensorflow.keras.optimizers.legacy as legacy_optimizers

# 数据路径
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
RESULTS_DIR = 'results'

# 文件名
TRAFFIC_FILE = 'traffic/metr-la.csv'
WEATHER_FILE = 'weather/noaa_weather_5min.csv'

# 数据处理参数
TRAIN_RATIO = 0.5
VAL_RATIO = 0.25
TEST_RATIO = 0.25

# 全局随机种子
RANDOM_SEED = 42

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 10,
    'verbose': 1,
    'callbacks': [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,
            min_lr=1e-4
        )
    ]
}

# 模型配置
MODEL_CONFIG = {
    'LSTM': {
        'units': [16],
        'dropout': 0.1,
        'l2_regularization': 1e-3,
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 0.01,
        'loss': 'mse',
        'metrics': ['mae']
    },
    'GRU': {
        'units': [16],
        'dropout': 0.1,
        'l2_regularization': 1e-3,
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 0.01,
        'loss': 'mse',
        'metrics': ['mae']
    },
    'CNN_LSTM': {
        'cnn_filters': [8],
        'cnn_kernel_size': 2,
        'lstm_units': [8],
        'dropout': 0.1,
        'l2_regularization': 1e-3,
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 0.01,
        'loss': 'mse',
        'metrics': ['mae']
    }
}

# 实验配置
EXPERIMENT_TYPES = ['baseline', 'enhanced']

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'style': 'seaborn',
    'color_palette': 'Set2',
    'dpi': 300
}
