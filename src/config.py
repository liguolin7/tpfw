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
TRAIN_RATIO = 0.7    # 训练集比例
VAL_RATIO = 0.15     # 验证集比例
TEST_RATIO = 0.15    # 测试集比例

# 全局随机种子
RANDOM_SEED = 42

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 200,
    'verbose': 1,
    'callbacks': [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_loss'
        )
    ]
}

# 模型配置
MODEL_CONFIG = {
    'LSTM': {
        'units': [256, 128, 64],
        'dropout': 0.2,
        'l2_regularization': 1e-6,
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 1e-3,
        'loss': 'huber',
        'metrics': ['mae', 'mse']
    },
    'GRU': {
        'units': [256, 128, 64],
        'dropout': 0.2,
        'l2_regularization': 1e-6,
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 1e-3,
        'loss': 'huber',
        'metrics': ['mae', 'mse']
    },
    'CNN_LSTM': {
        'cnn_filters': [128, 64],
        'cnn_kernel_size': 3,
        'lstm_units': [128, 64],
        'dropout': 0.2,
        'l2_regularization': 1e-6,
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 1e-3,
        'loss': 'huber',
        'metrics': ['mae', 'mse']
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

# 数据处理参数验证
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-10, "数据集划分比例之和必须等于1"
