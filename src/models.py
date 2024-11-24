from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

class BaselineModels:
    def __init__(self):
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lstm_model = None
        
    def create_lstm_model(self, input_shape):
        """创建LSTM模型"""
        model = Sequential([
            LSTM(50, input_shape=input_shape, return_sequences=True),
            LSTM(30),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.lstm_model = model
        
    # 添加训练和预测方法 