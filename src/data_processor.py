import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, 
    RANDOM_SEED, PROCESSED_DATA_DIR
)
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
        
        # 使用 IQR 方法检测和处理异常值
        Q1 = df['avg_speed'].quantile(0.25)
        Q3 = df['avg_speed'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 过滤在范围之外的异常值
        df = df[(df['avg_speed'] >= lower_bound) & (df['avg_speed'] <= upper_bound)]
        
        return df
        
    def process_weather_data(self, df):
        """处理天气数据"""
        logging.info("开始处理天气数据...")
        
        # 分析缺失值模式
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        logging.info(f"天气数据总缺失值数: {total_missing}")
        logging.info(f"每列缺失值概览:\n{missing_summary}")
        
        # 可视化缺失值模式（可选）
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # sns.heatmap(df.isnull(), cbar=False)
        # plt.show()

        # 视情况选择适当的填充方法
        # 例如，对于某些特征使用前向填充，对于其他特征使用均值填充
        df_filled = df.copy()
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype == 'float':
                    # 连续型变量使用插值法
                    df_filled[column] = df[column].interpolate(method='time')
                else:
                    # 类别型变量使用前向填充
                    df_filled[column] = df[column].fillna(method='ffill')
        
        # 如果仍有缺失值，使用后向填充
        df_filled = df_filled.fillna(method='bfill')
        
        # 检查是否还有缺失值
        remaining_missing = df_filled.isnull().sum().sum()
        if remaining_missing > 0:
            logging.warning(f"填充后仍有缺失值数: {remaining_missing}")
        else:
            logging.info("天气数据缺失值处理完成")
        
        # 后续处理...
        df = df_filled
        
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
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
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
        """创建特征"""
        # 确保数据按时间排序
        df = df.sort_index()
        
        # 创建时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # 添加节假日标记
        # 假设使用美国的公共假期
        import holidays
        us_holidays = holidays.US()
        df['is_holiday'] = df.index.to_series().apply(lambda x: 1 if x in us_holidays else 0)
        
        # 创建滞后特征
        df['speed_lag1'] = df['avg_speed'].shift(1)
        df['speed_lag2'] = df['avg_speed'].shift(2)
        df['speed_lag3'] = df['avg_speed'].shift(3)
        
        # 创建更长的移动平均特征
        df['speed_ma5'] = df['avg_speed'].rolling(window=5).mean()
        df['speed_ma10'] = df['avg_speed'].rolling(window=10).mean()
        df['speed_ma15'] = df['avg_speed'].rolling(window=15).mean()
        df['speed_ma30'] = df['avg_speed'].rolling(window=30).mean()
        
        # 天气类型编码
        if include_weather:
            # 假设天气数据中有天气描述字段，例如'weather_description'
            # 需要将天气描述进行类别编码
            if 'weather_description' in df.columns:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df['weather_encoded'] = le.fit_transform(df['weather_description'])
        
        # 移除包含NaN的行
        df = df.dropna()
        
        return df
        
    def split_data(self, data):
        """分割数据集"""
        # 提取目标变量
        y = data['target']
        # 删除目标变量列
        X = data.drop('target', axis=1)
        
        # 计算分割点
        train_size = int(len(data) * 0.5)
        val_size = int(len(data) * 0.25)
        
        # 分割数据集
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def prepare_data(self, traffic_data, weather_data=None):
        """准备实验数据"""
        # 基础数据处理
        processed_data = traffic_data.copy()
        
        # 设置目标变量（假设最后一列是目标变量）
        processed_data['target'] = processed_data.iloc[:, -1]
        
        if weather_data is not None:
            # 合并天气数据
            processed_data = pd.concat([processed_data, weather_data], axis=1)
        
        # 数据清洗
        processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
        processed_data = processed_data.fillna(method='ffill')
        processed_data = processed_data.fillna(method='bfill')
        
        # 标准化
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(processed_data),
            columns=processed_data.columns,
            index=processed_data.index
        )
        
        return scaled_data
        
    def prepare_sequences(self, traffic_data, weather_data=None):
        """准备序列数据
        
        Args:
            traffic_data: 交通数据
            weather_data: 天气数据（可选）
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        try:
            # 处理交通数据
            traffic_processed = self.process_traffic_data(traffic_data)
            
            if weather_data is not None:
                # 处理天气数据
                weather_processed = self.process_weather_data(weather_data)
                # 合并数据
                data = self.align_and_merge_data(traffic_processed, weather_processed)
            else:
                data = traffic_processed
                
            # 创建特征
            data = self.create_features(data, include_weather=(weather_data is not None))
            
            # 标准化数据
            data_scaled = self.prepare_data(data)
            
            # 分割数据集
            total_samples = len(data_scaled)
            train_size = int(total_samples * 0.5)
            val_size = int(total_samples * 0.25)
            
            # 准备特征和目标变量
            X = data_scaled.drop('target', axis=1)
            y = data_scaled['target']
            
            # 分割数据
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            
            X_test = X[train_size+val_size:]
            y_test = y[train_size+val_size:]
            
            logging.info("数据序列准备完成")
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logging.error(f"准备数据序列时出错: {str(e)}")
            raise