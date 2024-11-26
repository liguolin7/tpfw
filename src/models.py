import os
import tensorflow as tf
import numpy as np
import logging
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    GRU, Input, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Conv1D, MaxPooling1D
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, AdamW, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score
from tqdm import tqdm
from config import MODEL_CONFIG, TRAINING_CONFIG, RANDOM_SEED
from tensorflow.keras.regularizers import l2
from kerastuner import HyperModel
from sklearn.ensemble import RandomForestRegressor
import tensorflow.keras.optimizers.legacy as legacy_optimizers

class BaselineModels:
    def __init__(self):
        self.models = {}
        # 设置环境变量抑制优化器警告
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
        
        # 从 TRAINING_CONFIG 中移除 validation_data
        train_config = TRAINING_CONFIG.copy()
        if 'validation_data' in train_config:
            del train_config['validation_data']
        
        # 在 fit 调用时单独设置 validation_data
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
    
    def _train_model(self, model_type, X_train, y_train, X_val, y_val):
        """通用模型训练方法"""
        train_config = TRAINING_CONFIG
        
        # 创建模型（如果还没有创建）
        if self.models[model_type] is None:
            create_func = getattr(self, f'create_{model_type}_model')
            self.models[model_type] = create_func((X_train.shape[1], 1))
        
        # 创建回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(**train_config['early_stopping']),
            tf.keras.callbacks.ReduceLROnPlateau(**train_config['lr_scheduler'])
        ]
        
        # 训练模型
        history = self.models[model_type].fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=train_config['epochs'],
            batch_size=train_config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def create_lstm_model(self, input_shape):
        """创建LSTM模型"""
        config = MODEL_CONFIG['LSTM']
        
        try:
            tf.keras.utils.set_random_seed(RANDOM_SEED)
            model = Sequential([
                LSTM(config['units'][0],
                     input_shape=input_shape,
                     return_sequences=True,
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)),
                Dropout(config['dropout']),
                LSTM(config['units'][0],
                     return_sequences=False,
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)),
                Dense(1)
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss=config['loss'],
                metrics=config['metrics']
            )
            
            return model
                
        except Exception as e:
            logging.error(f"创建LSTM模型时出错: {str(e)}")
            raise
        
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
        
    def predict(self, model_name, X):
        """使用指定模型进行预测"""
        try:
            if model_name.lower() not in self.models:
                raise ValueError(f"Model {model_name} not found")
            return self.models[model_name.lower()].predict(X)
        except Exception as e:
            logging.error(f"预测时出错: {str(e)}")
            raise
    
    def create_gru_model(self, input_shape):
        """创建GRU模型"""
        config = MODEL_CONFIG['GRU']
        
        try:
            tf.keras.utils.set_random_seed(RANDOM_SEED)
            model = Sequential([
                GRU(config['units'][0],
                    input_shape=input_shape,
                    return_sequences=True,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)),
                Dropout(config['dropout']),
                GRU(config['units'][0],
                    return_sequences=False,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)),
                Dense(1)
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
            model.compile(
                optimizer=optimizer,
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
        
        try:
            tf.keras.utils.set_random_seed(RANDOM_SEED)
            model = Sequential([
                # CNN层
                Conv1D(
                    filters=config['cnn_filters'][0],
                    kernel_size=config['cnn_kernel_size'],
                    activation='relu',
                    input_shape=input_shape,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)
                ),
                MaxPooling1D(pool_size=2),
                BatchNormalization(),
                
                Conv1D(
                    filters=config['cnn_filters'][1],
                    kernel_size=config['cnn_kernel_size'],
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)
                ),
                MaxPooling1D(pool_size=2),
                BatchNormalization(),
                
                # LSTM层
                LSTM(config['lstm_units'][0], return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)),
                Dropout(config['dropout']),
                BatchNormalization(),
                
                LSTM(config['lstm_units'][1], kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)),
                Dropout(config['dropout']),
                BatchNormalization(),
                
                Dense(1)
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss=config['loss'],
                metrics=config['metrics']
            )
            
            return model
                
        except Exception as e:
            logging.error(f"创建CNN-LSTM模型时出错: {str(e)}")
            raise
    
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

    def train_random_forest(self, X_train, y_train):
        """训练随机森林模型"""
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    
    def _build_lstm_model(self, input_shape):
        """构建LSTM模型"""
        config = MODEL_CONFIG['LSTM']
        
        model = Sequential([
            # 第一层LSTM
            LSTM(config['units'][0], 
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            # 第二层LSTM
            LSTM(config['units'][1], 
                return_sequences=True,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            # 第三层LSTM
            LSTM(config['units'][2]),
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
            # 第一层GRU
            GRU(config['units'][0], 
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            # 第二层GRU
            GRU(config['units'][1], 
                return_sequences=True,
                kernel_regularizer=l2(config['l2_regularization'])),
            BatchNormalization(),
            Dropout(config['dropout']),
            
            # 第三层GRU
            GRU(config['units'][2]),
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

class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        # 调整层数和神经元数量
        for i in range(hp.Int('layers', 1, 3)):
            model.add(LSTM(
                units=hp.Int(f'units_{i}', min_value=16, max_value=128, step=16),
                input_shape=self.input_shape if i == 0 else None,
                return_sequences=False if i == hp.Int('layers', 1, 3) - 1 else True,
                kernel_regularizer=l2(hp.Float('l2', 1e-4, 1e-2, sampling='log'))
            ))
            model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        model.add(Dense(1))
        optimizer = AdamW(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
