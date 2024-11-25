import os
import tensorflow as tf
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    GRU, Input, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D
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
        self.transformer_model = None
        
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
        
    def create_transformer_model(self, input_shape):
        """创建改进的Transformer模型"""
        config = MODEL_CONFIG['Transformer']
        
        inputs = Input(shape=input_shape)
        
        # 添加位置编码
        pos_encoding = self.positional_encoding(input_shape[0], input_shape[1])
        x = inputs + pos_encoding
        
        # 多层Transformer块
        for _ in range(3):  # 增加层数
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=config['num_heads'],
                key_dim=config['head_size']
            )(x, x)
            
            # Add & Norm
            x = LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed Forward Network
            ffn = Sequential([
                Dense(config['ff_dim'], activation='relu'),
                Dropout(config['dropout']),
                Dense(input_shape[-1])
            ])
            
            ffn_output = ffn(x)
            x = LayerNormalization(epsilon=1e-6)(ffn_output + x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        x = Dropout(config['dropout'])(x)
        
        # Final dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(config['dropout'])(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss=config['loss'],
            metrics=config['metrics']
        )
        
        return model
        
    def positional_encoding(self, length, depth):
        """添加位置编码"""
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth
        
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1
        )
        
        return tf.cast(pos_encoding, dtype=tf.float32)
        
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
