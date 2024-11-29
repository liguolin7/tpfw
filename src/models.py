import os
import tensorflow as tf
import numpy as np
import logging
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    GRU, Conv1D, MaxPooling1D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import tensorflow.keras.optimizers.legacy as legacy_optimizers
from config import MODEL_CONFIG, TRAINING_CONFIG, RANDOM_SEED

class BaselineModels:
    def __init__(self):
        self.models = {}
        os.environ['TF_KERAS_BACKEND_LEGACY_OPTIMIZER'] = '1'
        os.environ['TF_KERAS_BACKEND_LEGACY_WARNING'] = '0'
    
    def get_model(self, model_name):
        """获取指定名称的模型"""
        if model_name.lower() not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        return self.models[model_name.lower()]
    
    def train_lstm(self, X_train, y_train, X_val, y_val):
        """训练LSTM模型"""
        model = self._build_lstm_model(X_train.shape[1:])
        
        train_config = TRAINING_CONFIG.copy()
        if 'validation_data' in train_config:
            del train_config['validation_data']
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            **train_config
        )
        self.models['lstm'] = model
        return history
    
    def train_gru(self, X_train, y_train, X_val, y_val):
        """训练GRU模型"""
        model = self._build_gru_model(X_train.shape[1:])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            **TRAINING_CONFIG
        )
        self.models['gru'] = model
        return history
    
    def train_cnn_lstm(self, X_train, y_train, X_val, y_val):
        """训练CNN-LSTM模型"""
        model = self._build_cnn_lstm_model(X_train.shape[1:])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            **TRAINING_CONFIG
        )
        self.models['cnn_lstm'] = model
        return history
    
    def predict(self, model_name, X):
        """使用指定模型进行预测"""
        try:
            if model_name.lower() not in self.models:
                raise ValueError(f"Model {model_name} not found")
            return self.models[model_name.lower()].predict(X)
        except Exception as e:
            logging.error(f"预测时出错: {str(e)}")
            raise
            
    def _build_lstm_model(self, input_shape):
        """构建LSTM模型"""
        config = MODEL_CONFIG['LSTM']
        
        model = Sequential([
            LSTM(config['units'][0], return_sequences=True, input_shape=input_shape),
            LSTM(config['units'][1], return_sequences=True),
            LSTM(config['units'][-1]),
            Dense(1)
        ])
        
        optimizer = legacy_optimizers.Adam(
            learning_rate=config['learning_rate'],
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=config['loss'],
            metrics=config['metrics']
        )
        
        return model
    
    def _build_gru_model(self, input_shape):
        """构建GRU模型"""
        config = MODEL_CONFIG['GRU']
        
        model = Sequential([
            GRU(config['units'][0], 
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            GRU(config['units'][1], 
                return_sequences=False,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])
        
        optimizer = legacy_optimizers.Adam(
            learning_rate=config['learning_rate'],
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=config['loss'],
            metrics=config['metrics']
        )
        
        return model
    
    def _build_cnn_lstm_model(self, input_shape):
        """构建CNN-LSTM模型"""
        config = MODEL_CONFIG['CNN_LSTM']
        
        model = Sequential([
            Conv1D(
                filters=config['cnn_filters'][0],
                kernel_size=config['cnn_kernel_size'],
                activation='relu',
                input_shape=input_shape,
                kernel_regularizer=l2(config['l2_regularization'])
            ),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(
                filters=config['cnn_filters'][1],
                kernel_size=config['cnn_kernel_size'],
                activation='relu',
                kernel_regularizer=l2(config['l2_regularization'])
            ),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            LSTM(config['lstm_units'][0], 
                return_sequences=True,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            LSTM(config['lstm_units'][1]),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])
        
        optimizer = legacy_optimizers.Adam(
            learning_rate=config['learning_rate'],
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=config['loss'],
            metrics=config['metrics']
        )
        
        return model
