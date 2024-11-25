import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, 
    RANDOM_STATE, TRAINING_CONFIG
)
import logging
from tqdm import tqdm

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def process_traffic_data(self, df):
        """处理交通数据
        
        Args:
            df (pd.DataFrame): 原始交通数据
        
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 将时间索引转换为datetime
        df.index = pd.to_datetime(df.index)
        
        # 计算所有传感器的平均速度
        df['avg_speed'] = df.mean(axis=1)
        
        # 移除空值
        df = df.dropna()
        
        # 移除异常值 (比如速度为0或异常高的值)
        df = df[(df['avg_speed'] > 0) & (df['avg_speed'] < 100)]
        
        return df
        
    def process_weather_data(self, df):
        """处理天气数据"""
        logging.info("开始处理天气数据...")
        
        # 检查缺失值
        missing_columns = df.isnull().sum()[df.isnull().sum() > 0].shape[0]
        if missing_columns > 0:
            logging.info(f"处理{missing_columns}个含缺失值的天气特征...")
        
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
            weather_df: 理后的天气数据
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
        
    def create_features(self, df, include_weather=True):
        """创建特征
        
        Args:
            df (pd.DataFrame): 预处理后的数据
            include_weather (bool): 是否包含天气特征
        
        Returns:
            pd.DataFrame: 包含新特征的数据框
        """
        # 确保数据按时间排序
        df = df.sort_index()
        
        # 创建时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # 创建滞后特征
        df['speed_lag1'] = df['avg_speed'].shift(1)
        df['speed_lag2'] = df['avg_speed'].shift(2)
        df['speed_lag3'] = df['avg_speed'].shift(3)
        
        # 创建移动平均特征
        df['speed_ma5'] = df['avg_speed'].rolling(window=5).mean()
        df['speed_ma10'] = df['avg_speed'].rolling(window=10).mean()
        
        # 移除包含NaN的行
        df = df.dropna()
        
        return df
        
    def split_data(self, df, target_col='avg_speed'):
        """划分数据集"""
        logging.info("开始划分数据集...")
        
        # 移除数据量限制，使用全部数据
        # 使用时间序列分割
        train_size = int(len(df) * TRAIN_RATIO)
        val_size = int(len(df) * VAL_RATIO)
        
        # 分离特征和目标变量
        y = df[target_col].values
        X = df.drop(columns=[target_col])
        
        # 快速分割
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # 使用更快的标准化方法
        self.scaler.fit(X_train)
        X_train = pd.DataFrame(self.scaler.transform(X_train), columns=X.columns, index=X_train.index)
        X_val = pd.DataFrame(self.scaler.transform(X_val), columns=X.columns, index=X_val.index)
        X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X.columns, index=X_test.index)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def prepare_data(self, traffic_data, weather_data=None):
        """准备实验数据
        
        Args:
            traffic_data (pd.DataFrame): 交通数据
            weather_data (pd.DataFrame, optional): 天气数据
            
        Returns:
            pd.DataFrame: 处理后的数据集
        """
        logging.info("准备实验数据...")
        processed_traffic = self.process_traffic_data(traffic_data)
        
        if weather_data is not None:
            processed_weather = self.process_weather_data(weather_data)
            merged_df = self.align_and_merge_data(processed_traffic, processed_weather)
            return self.create_features(merged_df, include_weather=True)
        
        return self.create_features(processed_traffic, include_weather=False)