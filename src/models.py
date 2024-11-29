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
    """基准模型类，只使用交通数据"""
    
    def __init__(self):
        self.models = {}
        
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """训练指定的模型"""
        if model_name == 'LSTM':
            model = self._build_lstm_model(X_train.shape[1:])
        elif model_name == 'GRU':
            model = self._build_gru_model(X_train.shape[1:])
        elif model_name == 'CNN_LSTM':
            model = self._build_cnn_lstm_model(X_train.shape[1:])
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
            
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=TRAINING_CONFIG['batch_size'],
            epochs=TRAINING_CONFIG['epochs'],
            callbacks=TRAINING_CONFIG['callbacks'],
            verbose=TRAINING_CONFIG['verbose']
        )
        
        self.models[model_name] = model
        return model, history
        
    def _build_lstm_model(self, input_shape):
        """构建LSTM模型"""
        config = MODEL_CONFIG['LSTM']
        
        model = Sequential([
            LSTM(config['units'][0], 
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            LSTM(config['units'][1],
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
        """构建CNN-LSTM混合模型"""
        config = MODEL_CONFIG['CNN_LSTM']
        
        model = Sequential([
            # CNN层
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
            
            # LSTM层
            LSTM(
                config['lstm_units'][0],
                return_sequences=True,
                kernel_regularizer=l2(config['l2_regularization'])
            ),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            LSTM(
                config['lstm_units'][1],
                return_sequences=False,
                kernel_regularizer=l2(config['l2_regularization'])
            ),
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

class EnhancedModels:
    """增强模型类，使用交通数据和天气数据"""
    
    def __init__(self):
        self.models = {}
        
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """训练指定的模型"""
        if model_name == 'LSTM':
            model = self._build_lstm_model(X_train.shape[1:])
        elif model_name == 'GRU':
            model = self._build_gru_model(X_train.shape[1:])
        elif model_name == 'CNN_LSTM':
            model = self._build_cnn_lstm_model(X_train.shape[1:])
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
            
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=TRAINING_CONFIG['batch_size'],
            epochs=TRAINING_CONFIG['epochs'],
            callbacks=TRAINING_CONFIG['callbacks'],
            verbose=TRAINING_CONFIG['verbose']
        )
        
        self.models[model_name] = model
        return model, history
    
    def _build_lstm_model(self, input_shape):
        """构建增强版LSTM模型"""
        config = MODEL_CONFIG['LSTM']
        
        model = Sequential([
            LSTM(config['units'][0], 
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            LSTM(config['units'][1],
                return_sequences=False,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            Dense(64, activation='relu'),  # 增加神经元数量
            BatchNormalization(),
            Dense(32, activation='relu'),  # 添加一层
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
    
    def _build_gru_model(self, input_shape):
        """构建增强版GRU模型"""
        config = MODEL_CONFIG['GRU']
        
        model = Sequential([
            GRU(config['units'][0], 
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            GRU(config['units'][1],
                return_sequences=True,  # 添加一层GRU
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            GRU(config['units'][1],
                return_sequences=False,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
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
        """构建增强版CNN-LSTM混合模型"""
        config = MODEL_CONFIG['CNN_LSTM']
        
        model = Sequential([
            # 增强的CNN层
            Conv1D(
                filters=config['cnn_filters'][0] * 2,  # 增加滤波器数量
                kernel_size=config['cnn_kernel_size'],
                activation='relu',
                input_shape=input_shape,
                kernel_regularizer=l2(config['l2_regularization'])
            ),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(
                filters=config['cnn_filters'][1] * 2,
                kernel_size=config['cnn_kernel_size'],
                activation='relu',
                kernel_regularizer=l2(config['l2_regularization'])
            ),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # 增强的LSTM层
            LSTM(
                config['lstm_units'][0] * 2,
                return_sequences=True,
                kernel_regularizer=l2(config['l2_regularization'])
            ),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            LSTM(
                config['lstm_units'][1] * 2,
                return_sequences=True,
                kernel_regularizer=l2(config['l2_regularization'])
            ),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            LSTM(
                config['lstm_units'][1],
                return_sequences=False,
                kernel_regularizer=l2(config['l2_regularization'])
            ),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
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
