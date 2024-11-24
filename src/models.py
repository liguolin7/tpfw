from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import logging
from config import RANDOM_STATE

class BaselineModels:
    def __init__(self):
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=RANDOM_STATE,
            n_jobs=-1  # 使用所有可用CPU核心
        )
        self.lstm_model = None
        
    def train_linear_regression(self, X_train, y_train):
        """训练线性回归模型"""
        logging.info("训练线性回归模型...")
        self.lr_model.fit(X_train, y_train)
        score = self.lr_model.score(X_train, y_train)
        logging.info(f"线性回归训练集 R2 分数: {score:.4f}")
        
    def train_random_forest(self, X_train, y_train):
        """训练随机森林模型"""
        logging.info("训练随机森林模型...")
        self.rf_model.fit(X_train, y_train)
        score = self.rf_model.score(X_train, y_train)
        logging.info(f"随机森林训练集 R2 分数: {score:.4f}")
        
    def create_lstm_model(self, input_shape):
        """创建LSTM模型"""
        logging.info(f"创建LSTM模型，输入形状: {input_shape}")
        model = Sequential([
            LSTM(50, input_shape=input_shape, return_sequences=True),
            LSTM(30),
            Dense(1)
        ])
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        self.lstm_model = model
        model.summary()
        return model
        
    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=1, batch_size=32):
        """训练LSTM模型"""
        if self.lstm_model is None:
            raise ValueError("LSTM模型未创建，请先调用create_lstm_model")
            
        logging.info("训练LSTM模型...")
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        logging.info("LSTM模型训练完成")
        return history
        
    def predict_linear_regression(self, X):
        """线性回归模型预测"""
        if self.lr_model is None:
            raise ValueError("线性回归模型未训练")
        return self.lr_model.predict(X)
        
    def predict_random_forest(self, X):
        """随机森林模型预测"""
        if self.rf_model is None:
            raise ValueError("随机森林模型未训练")
        return self.rf_model.predict(X)
        
    def predict_lstm(self, X):
        """LSTM模型预测"""
        if self.lstm_model is None:
            raise ValueError("LSTM模型未训练")
        return self.lstm_model.predict(X).flatten()  # 展平预测结果