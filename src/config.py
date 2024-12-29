# 配置文件路径和模型参数
import os
import tensorflow as tf
import tensorflow.keras.optimizers.legacy as legacy_optimizers
import importlib

# 数据路径
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
RESULTS_DIR = 'results'

# 文件名
TRAFFIC_FILE = 'traffic/metr-la.csv'
WEATHER_FILE = 'weather/noaa_weather_5min.csv'

# 数据处理参数
TRAIN_RATIO = 0.7    # 调整训练集比例
VAL_RATIO = 0.15     # 增加验证集比例
TEST_RATIO = 0.15    # 增加测试集比例

# 全局随机种子
RANDOM_SEED = 42

# 创建一个配置类来存储训练参数
class TrainingConfig:
    def __init__(self):
        self.config = {
            'batch_size': 64,        # 增加batch size以提高训练稳定性
            'epochs': 150,           # 增加训练轮数
            'verbose': 1,
            'callbacks': [
                # 更温和的学习率预热和衰减策略
                tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: 1e-4 * tf.math.exp(0.05 * epoch) if epoch < 20
                    else 1e-4 * tf.math.exp(-0.05 * (epoch - 20))
                ),
                
                # 更温和的ReduceLROnPlateau
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,       # 更温和的学习率衰减
                    patience=15,       # 增加耐心值
                    min_lr=1e-7,
                    verbose=1
                ),
                
                # 调整EarlyStopping
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=30,       # 增加耐心值
                    restore_best_weights=True,
                    min_delta=1e-5,    # 降低最小改善阈值
                    verbose=1
                ),
                
                # 保持ModelCheckpoint不变
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='best_model.h5',
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                )
            ]
        }

# 创建全局配置实例
training_config = TrainingConfig()

def get_training_config():
    """获取训练配置"""
    # 强制重新加载模块
    importlib.reload(tf.keras.callbacks)
    return training_config.config

def get_model_config():
    """获取模型配置"""
    return {
        'LSTM': {
            'units': [128, 64, 32],      # 减小网络规模，避免过拟合
            'dropout': 0.2,              # 减小dropout
            'l2_regularization': 1e-6,   # 减小正则化强度
            'optimizer': legacy_optimizers.Adam,
            'learning_rate': 1e-4,       # 降低学习率
            'loss': 'mse',               # 使用MSE损失
            'metrics': ['mae', 'mse', 'mape']
        },
        'GRU': {
            'units': [128, 64, 32],
            'dropout': 0.2,
            'l2_regularization': 1e-6,
            'optimizer': legacy_optimizers.Adam,
            'learning_rate': 1e-4,
            'loss': 'mse',
            'metrics': ['mae', 'mse', 'mape']
        },
        'CNN_LSTM': {
            'cnn_filters': [64, 32, 16],  # 减小CNN复杂度
            'cnn_kernel_size': 3,         # 减小卷积核尺寸
            'lstm_units': [64, 32, 16],   # 减小LSTM单元数
            'dropout': 0.2,
            'l2_regularization': 1e-6,
            'optimizer': legacy_optimizers.Adam,
            'learning_rate': 1e-4,
            'loss': 'mse',
            'metrics': ['mae', 'mse', 'mape']
        }
    }

# 验置
EXPERIMENT_TYPES = ['baseline', 'enhanced']

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'style': 'seaborn',
    'color_palette': 'Set2',
    'dpi': 300
}

# 数据处理参数验证
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-10, "数据集划分之和必须等于1"

# 添加数据处理配置
DATA_CONFIG = {
    'sequence_length': 12,        # 减小序列长度，避免噪声影响
    'prediction_horizon': 3,      # 保持预测步长不变
    'weather_feature_selection': {
        'correlation_threshold': 0.1,  # 提高相关性阈值，只保留重要特征
        'importance_threshold': 0.05   
    },
    'features': {
        'traffic': ['avg_speed', 'volume', 'occupancy'],
        'weather': [
            'TMAX', 'TMIN', 'PRCP', 'AWND',  # 减少天气特征，只保留最重要的
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
# 替换为动态导入函数
def get_config():
    """获取所有配置"""
    return {
        'training': get_training_config(),
        'model': get_model_config()
    }

