import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import *
import logging

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def process_traffic_data(self, df):
        """处理交通数据
        
        Args:
            df: 原始交通数据DataFrame
        Returns:
            处理后的交通数据DataFrame
        """
        logging.info("开始处理交通数据...")
        
        # 检查缺失值
        missing_stats = df.isnull().sum()
        logging.info(f"缺失值统计:\n{missing_stats}")
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 添加时间特征
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        
        # 计算每个传感器的平均速度
        df['avg_speed'] = df.mean(axis=1)
        
        logging.info("交通数据处理完成")
        return df
        
    def process_weather_data(self, df):
        """处理天气数据
        
        Args:
            df: 原始天气数据DataFrame
        Returns:
            处理后的天气数据DataFrame
        """
        logging.info("开始处理天气数据...")
        
        # 检查缺失值
        missing_stats = df.isnull().sum()
        logging.info(f"天气数据缺失值统计:\n{missing_stats}")
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 选择重要的天气特征
        selected_features = [
            'TMAX', 'TMIN',  # 温度
            'PRCP',         # 降水量
            'AWND',         # 风速
            'RHAV',         # 相对湿度
            'ASLP'          # 气压
        ]
        
        df = df[selected_features]
        
        # 添加衍生特征
        df['temp_diff'] = df['TMAX'] - df['TMIN']  # 温差
        df['is_raining'] = (df['PRCP'] > 0).astype(int)  # 是否有降水
        
        # 处理异常值
        for col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        logging.info("天气数据处理完成")
        return df
        
    def align_and_merge_data(self, traffic_df, weather_df):
        """对齐并合并交通和天气数据
        
        Args:
            traffic_df: 处理后的交通数据
            weather_df: 处理后的天气数据
        Returns:
            合并后的DataFrame
        """
        logging.info("开始合并数据...")
        
        # 确保时间频率一致（使用5分钟间隔）
        traffic_df = traffic_df.resample('5T').mean()
        weather_df = weather_df.resample('5T').mean()
        
        # 对齐时间索引
        common_idx = traffic_df.index.intersection(weather_df.index)
        traffic_df = traffic_df.loc[common_idx]
        weather_df = weather_df.loc[common_idx]
        
        # 合并数据
        merged_df = pd.concat([traffic_df, weather_df], axis=1)
        
        # 删除包含缺失值的行
        merged_df = merged_df.dropna()
        
        logging.info(f"数据合并完成，最终数据形状: {merged_df.shape}")
        return merged_df
        
    def create_features(self, df):
        """创建特征
        
        Args:
            df: 合并后的DataFrame
        Returns:
            添加特征后的DataFrame
        """
        logging.info("开始特征工程...")
        
        # 创建滞后特征（前一个时间点的速度）
        df['speed_lag1'] = df['avg_speed'].shift(1)
        
        # 创建移动平均特征
        df['speed_ma5'] = df['avg_speed'].rolling(window=5).mean()
        df['speed_ma10'] = df['avg_speed'].rolling(window=10).mean()
        
        # 创建时间特征
        df['is_rush_hour'] = df['hour'].apply(
            lambda x: 1 if (x >= 7 and x <= 9) or (x >= 16 and x <= 18) else 0
        )
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # 删除创建特征过程中产生的缺失值
        df = df.dropna()
        
        logging.info("特征工程完成")
        return df
        
    def split_data(self, df, target_col='avg_speed'):
        """划分数据集
        
        Args:
            df: 特征工程后的DataFrame
            target_col: 目标变量列名
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logging.info("开始划分数据集...")
        
        # 计算划分点
        n = len(df)
        train_size = int(n * TRAIN_RATIO)
        val_size = int(n * VAL_RATIO)
        
        # 分离特征和目标变量
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # 按时间顺序划分数据集
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # 标准化特征
        numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
        X_train[numeric_features] = self.scaler.fit_transform(X_train[numeric_features])
        X_val[numeric_features] = self.scaler.transform(X_val[numeric_features])
        X_test[numeric_features] = self.scaler.transform(X_test[numeric_features])
        
        logging.info(f"数据集划分完成:")
        logging.info(f"训练集大小: {len(X_train)}")
        logging.info(f"验证集大小: {len(X_val)}")
        logging.info(f"测试集大小: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test