import os
import tensorflow as tf
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    GRU, Input, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Conv1D, MaxPooling1D
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from tqdm import tqdm
from config import RANDOM_STATE, MODEL_CONFIG, TRAINING_CONFIG

class BaselineModels:
    def __init__(self):
        self.lstm_model = None
        self.gru_model = None
        self.cnn_lstm_model = None
        
    def create_lstm_model(self, input_shape):
        """创建更复杂的LSTM模型"""
        config = MODEL_CONFIG['LSTM']
        
        model = Sequential([
            LSTM(config['units'][0], 
                 input_shape=input_shape, 
                 return_sequences=True),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            LSTM(config['units'][1], 
                 return_sequences=True),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            LSTM(config['units'][2], 
                 return_sequences=False),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss=config['loss'],
            metrics=config['metrics']
        )
        
        return model
        
    def create_custom_callback(self, model_name):
        """创建自定义回调来优化输出"""
        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                logging.info(f"\nEpoch {epoch+1}/{self.params['epochs']}")
                
            def on_epoch_end(self, epoch, logs=None):
                metrics_str = " - ".join([
                    f"{k}: {v:.4f}" for k, v in logs.items()
                ])
                logging.info(f"{model_name} - {metrics_str}")
        
        return CustomCallback()
        
    def train_lstm(self, X_train, y_train, X_val, y_val):
        """训练LSTM模型"""
        train_config = TRAINING_CONFIG
        
        # 创建模型（如果还没有创建）
        if self.lstm_model is None:
            self.lstm_model = self.create_lstm_model((X_train.shape[1], 1))
        
        # 创建回调函数，减少输出
        callbacks = [
            tf.keras.callbacks.EarlyStopping(**train_config['early_stopping']),
            tf.keras.callbacks.ReduceLROnPlateau(**train_config['lr_scheduler'])
        ]
        
        # 设置训练时的verbose级别
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=train_config['max_epochs'],
            batch_size=train_config['batch_size'],
            callbacks=callbacks,
            verbose=1  # 可以改为0来完全禁用进度条
        )
        
        return history
        
    def predict_lstm(self, X):
        """LSTM模型预测"""
        if self.lstm_model is None:
            raise ValueError("LSTM模型未训练")
        
        # X已经是numpy数组，不需要调用.values
        if isinstance(X, np.ndarray):
            X_reshaped = X  # 已经是正确的形状
        else:
            X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
        
        return self.lstm_model.predict(X_reshaped).flatten()
        
    def create_gru_model(self, input_shape):
        """创建GRU模型"""
        config = MODEL_CONFIG['GRU']
        
        try:
            with tf.device('/CPU:0'):
                model = Sequential([
                    GRU(config['units'][0], 
                        input_shape=input_shape, 
                        return_sequences=True),
                    Dropout(config['dropout']),
                    GRU(config['units'][1], 
                        return_sequences=False),
                    Dense(1)
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=config['learning_rate']),
                    loss=config['loss'],
                    metrics=config['metrics']
                )
                
                return model
                
        except Exception as e:
            logging.error(f"创建GRU模型时出错: {str(e)}")
            raise
        
    def create_cnn_lstm_model(self, input_shape):
        """创建CNN-LSTM混合模型"""
        config = MODEL_CONFIG['CNN_LSTM']
        
        model = Sequential([
            # CNN层
            Conv1D(
                filters=config['cnn_filters'][0],
                kernel_size=config['cnn_kernel_size'],
                activation='relu',
                input_shape=input_shape
            ),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            
            Conv1D(
                filters=config['cnn_filters'][1],
                kernel_size=config['cnn_kernel_size'],
                activation='relu'
            ),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            
            # LSTM层
            LSTM(config['lstm_units'][0], return_sequences=True),
            Dropout(config['dropout']),
            BatchNormalization(),
            
            LSTM(config['lstm_units'][1]),
            Dropout(config['dropout']),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss=config['loss'],
            metrics=config['metrics']
        )
        
        return model
    
    def train_cnn_lstm(self, X_train, y_train, X_val, y_val):
        """训练CNN-LSTM模型"""
        if self.cnn_lstm_model is None:
            self.cnn_lstm_model = self.create_cnn_lstm_model((X_train.shape[1], 1))
            
        train_config = TRAINING_CONFIG
        callbacks = [
            tf.keras.callbacks.EarlyStopping(**train_config['early_stopping']),
            tf.keras.callbacks.ReduceLROnPlateau(**train_config['lr_scheduler'])
        ]
        
        history = self.cnn_lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=train_config['max_epochs'],
            batch_size=train_config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_cnn_lstm(self, X):
        """CNN-LSTM模型预测"""
        if self.cnn_lstm_model is None:
            raise ValueError("CNN-LSTM模型未训练")
        
        if isinstance(X, np.ndarray):
            X_reshaped = X
        else:
            X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
        
        return self.cnn_lstm_model.predict(X_reshaped).flatten()
        
    def train_gru(self, X_train, y_train, X_val, y_val):
        """训练GRU模型"""
        train_config = TRAINING_CONFIG
        
        # 创建模型（如果还没有创建）
        if self.gru_model is None:
            self.gru_model = self.create_gru_model((X_train.shape[1], 1))
        
        # 创建回调函数
        early_stopping = tf.keras.callbacks.EarlyStopping(**train_config['early_stopping'])
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(**train_config['lr_scheduler'])
        
        # 训练模型
        history = self.gru_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=train_config['max_epochs'],
            batch_size=train_config['batch_size'],
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
        return history
        
    def train_transformer(self, X_train, y_train, X_val, y_val):
        """训练Transformer模型"""
        train_config = TRAINING_CONFIG
        
        # 创建模型（如果还没有创建）
        if self.transformer_model is None:
            self.transformer_model = self.create_transformer_model((X_train.shape[1], 1))
        
        # 创建回调函数
        early_stopping = tf.keras.callbacks.EarlyStopping(**train_config['early_stopping'])
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(**train_config['lr_scheduler'])
        
        # 训练模型
        history = self.transformer_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=train_config['max_epochs'],
            batch_size=train_config['batch_size'],
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
        return history
        
    def predict_gru(self, X):
        """GRU模型预测"""
        if self.gru_model is None:
            raise ValueError("GRU模型未训练")
        
        # X已经是numpy数组，不需要调用.values
        if isinstance(X, np.ndarray):
            X_reshaped = X  # 已经是正确的形状
        else:
            X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
        
        return self.gru_model.predict(X_reshaped).flatten()
        
    def predict_transformer(self, X):
        """Transformer模型预测"""
        if self.transformer_model is None:
            raise ValueError("Transformer模型未训练")
        
        # X已经是numpy数组，不需要调用.values
        if isinstance(X, np.ndarray):
            X_reshaped = X  # 已经是正确的形状
        else:
            X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
        
        return self.transformer_model.predict(X_reshaped).flatten()
