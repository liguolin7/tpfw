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
    'batch_size': 128,        # 减小batch size以提高模型灵活性
    'epochs': 1,           # 设置训练轮数为1
    'verbose': 1,
    'callbacks': [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,          # 更温和的学习率衰减
            patience=5,          # 减小耐心值，更快响应
            min_lr=1e-6,
            verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=0,
            save_weights_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,         # 减小耐心值，及时停止
            restore_best_weights=True,
            min_delta=1e-5,
            verbose=0
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]
}

# 模型配置
MODEL_CONFIG = {
    'LSTM': {
        'units': [64, 32],      # 减小网络规模，避免过拟合
        'dropout': 0.2,         # 减小dropout
        'l2_regularization': 1e-4,  # 增加正则化强度
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 5e-4,   # 减小学习率
        'loss': 'mse',          # 使用标准MSE损失
        'metrics': ['mae', 'mse']
    },
    'GRU': {
        'units': [64, 32],
        'dropout': 0.2,
        'l2_regularization': 1e-4,
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 5e-4,
        'loss': 'mse',
        'metrics': ['mae', 'mse']
    },
    'CNN_LSTM': {
        'cnn_filters': [32, 16],  # 减小滤波器数量
        'cnn_kernel_size': 3,
        'lstm_units': [32, 16],   # 减小LSTM单元
        'dropout': 0.2,
        'l2_regularization': 1e-4,
        'optimizer': legacy_optimizers.Adam,
        'learning_rate': 5e-4,
        'loss': 'mse',
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
    'sequence_length': 12,        # 减小序列长度，聚焦于短期依赖
    'prediction_horizon': 1,
    'weather_feature_selection': {
        'correlation_threshold': 0.1,  # 提高阈值，只保留重要特征
        'importance_threshold': 0.05   
    },
    'features': {
        'traffic': ['avg_speed'],
        'weather': [
            'TMAX', 'TMIN', 'PRCP', 'AWND', 'RHAV',  # ���留基础特征
            'temp_range', 'feels_like',
            'severe_weather', 'rush_hour_rain'
        ]
    }
}

# 添加天气分析配置
WEATHER_ANALYSIS_CONFIG = {
    'extreme_weather_threshold': 0.85,  # 调整极端天气阈值
    'rush_hour_periods': [(6, 10), (16, 20)],  # 扩大高峰时段范围
    'weather_features': [
        'Temperature', 'Precipitation', 'Wind Speed', 'Humidity',
        'Wind Chill', 'Heat Index', 'Visibility', 'Pressure'
    ]
}
