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
TRAIN_RATIO = 0.8    # 增加训练集比例
VAL_RATIO = 0.1     
TEST_RATIO = 0.1     

# 全局随机种子
RANDOM_SEED = 42

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 32,     # 减小batch size以增加训练稳定性
    'epochs': 1,        # 设置为1
    'verbose': 1,
    'callbacks': [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,   # 增加耐心值
            restore_best_weights=True,
            min_delta=1e-4
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
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
        'units': [128, 64],
        'dropout': 0.3,       # 增加dropout
        'l2_regularization': 1e-5,  # 增加正则化
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 1e-3,  # 调整学习率
        'loss': 'huber',
        'metrics': ['mae', 'mse']
    },
    'GRU': {
        'units': [128, 64],
        'dropout': 0.2,
        'l2_regularization': 1e-6,
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 2e-3,
        'loss': 'huber',
        'metrics': ['mae', 'mse']
    },
    'CNN_LSTM': {
        'cnn_filters': [64, 32],
        'cnn_kernel_size': 3,
        'lstm_units': [64, 32],
        'dropout': 0.2,
        'l2_regularization': 1e-6,
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 2e-3,
        'loss': 'huber',
        'metrics': ['mae', 'mse']
    },
    'lstm': {
        'units': [64, 32, 16],
        'dropout': 0.2,
        'l2_regularization': 1e-6,
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 2e-3,
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

# 添加数据处理配置
DATA_CONFIG = {
    'sequence_length': 12,
    'prediction_horizon': 1,
    'weather_feature_selection': {
        'correlation_threshold': 0.1,  # 相关性阈值
        'importance_threshold': 0.01   # 特征重要性阈值
    },
    'features': {
        'traffic': ['avg_speed'],
        'weather': [
            'TMAX', 'TMIN', 'PRCP', 'AWND', 'RHAV',
            'temp_range', 'feels_like', 'wind_chill',
            'severe_weather', 'rush_hour_rain'
        ]
    }
}

# 添加天气分析配置
WEATHER_ANALYSIS_CONFIG = {
    'extreme_weather_threshold': 0.95,  # 极端天气阈值（95百分位）
    'rush_hour_periods': [(7, 9), (17, 19)],  # 早晚高峰时段
    'weather_features': ['Temperature', 'Precipitation', 'Wind Speed', 'Humidity']
}
