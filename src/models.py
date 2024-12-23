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
import warnings

class BaselineModels:
    """基准模型类 - 使用简单结构"""
    def __init__(self):
        self.models = {}
        
    def build_lstm(self, input_shape):
        """构建基础LSTM模型"""
        model = tf.keras.Sequential([
            # 添加输入标准化层
            tf.keras.layers.BatchNormalization(input_shape=input_shape),
            
            # 第一个LSTM层
            tf.keras.layers.LSTM(
                units=MODEL_CONFIG['LSTM']['units'][0],
                return_sequences=True,
                dropout=MODEL_CONFIG['LSTM']['dropout'],
                recurrent_dropout=0.1,  # 添加循环dropout
                kernel_regularizer=tf.keras.regularizers.l2(
                    MODEL_CONFIG['LSTM']['l2_regularization']
                )
            ),
            tf.keras.layers.BatchNormalization(),
            
            # 第二个LSTM层
            tf.keras.layers.LSTM(
                units=MODEL_CONFIG['LSTM']['units'][1],
                return_sequences=False,
                dropout=MODEL_CONFIG['LSTM']['dropout'],
                recurrent_dropout=0.1
            ),
            
            # 全连接层
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def build_gru(self, input_shape):
        """构建基础GRU模型"""
        model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(input_shape=input_shape),
            
            # 双向GRU
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                units=MODEL_CONFIG['GRU']['units'][0],
                return_sequences=True,
                dropout=MODEL_CONFIG['GRU']['dropout'],
                recurrent_dropout=0.1
            )),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.GRU(
                units=MODEL_CONFIG['GRU']['units'][1],
                return_sequences=False,
                dropout=MODEL_CONFIG['GRU']['dropout']
            ),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def build_cnn_lstm(self, input_shape):
        """构建基础CNN-LSTM模型"""
        model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(input_shape=input_shape),
            
            # 并行卷积层
            tf.keras.layers.Conv1D(
                filters=MODEL_CONFIG['CNN_LSTM']['cnn_filters'][0],
                kernel_size=3,
                padding='same',
                activation='relu'
            ),
            tf.keras.layers.Conv1D(
                filters=MODEL_CONFIG['CNN_LSTM']['cnn_filters'][0],
                kernel_size=5,
                padding='same',
                activation='relu'
            ),
            
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            
            # LSTM层
            tf.keras.layers.LSTM(
                units=MODEL_CONFIG['CNN_LSTM']['lstm_units'][0],
                return_sequences=True,
                dropout=MODEL_CONFIG['CNN_LSTM']['dropout']
            ),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.LSTM(
                units=MODEL_CONFIG['CNN_LSTM']['lstm_units'][1],
                return_sequences=False
            ),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """训练模型"""
        # 使用静默模式
        with tf.keras.utils.CustomObjectScope({}):
            # 使用上下文管理器来抑制警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # 使用 GPU 设备
                with tf.device('/GPU:0'):
                    if model_name not in self.models:
                        if model_name == 'LSTM':
                            model = self.build_lstm(X_train.shape[1:])
                        elif model_name == 'GRU':
                            model = self.build_gru(X_train.shape[1:])
                        elif model_name == 'CNN_LSTM':
                            model = self.build_cnn_lstm(X_train.shape[1:])
                        else:
                            raise ValueError(f"Unknown model name: {model_name}")
                        
                        # 编译模型
                        model.compile(
                            optimizer=MODEL_CONFIG[model_name]['optimizer'](
                                learning_rate=MODEL_CONFIG[model_name]['learning_rate']
                            ),
                            loss=MODEL_CONFIG[model_name]['loss'],
                            metrics=MODEL_CONFIG[model_name]['metrics']
                        )
                        
                        self.models[model_name] = model
                    
                    logging.info(f"Training {model_name} for {TRAINING_CONFIG['epochs']} epochs")
                    
                    # 训练模型
                    history = self.models[model_name].fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=TRAINING_CONFIG['batch_size'],
                        epochs=TRAINING_CONFIG['epochs'],
                        verbose=1,
                        callbacks=TRAINING_CONFIG['callbacks'],
                        use_multiprocessing=True,
                        workers=4
                    )
                    
                    return self.models[model_name], history

class EnhancedModels:
    """增强模型类 - 使用更复杂的结构处理天气特征"""
    def __init__(self):
        self.models = {}
        
    def build_lstm(self, input_shape):
        """构建增强LSTM模型"""
        # 分离输入
        inputs = tf.keras.Input(shape=input_shape)
        traffic_features = inputs[:, :, :207]  # 交通特征
        weather_features = inputs[:, :, 207:]  # 天气特征
        
        # 天气特征处理分支
        weather = tf.keras.layers.BatchNormalization()(weather_features)
        weather = tf.keras.layers.LSTM(
            units=32,
            return_sequences=True,
            dropout=0.2
        )(weather)
        
        # 交通特征处理分支
        traffic = tf.keras.layers.BatchNormalization()(traffic_features)
        traffic = tf.keras.layers.LSTM(
            units=64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1
        )(traffic)
        
        # 特征融合
        concat = tf.keras.layers.Concatenate()([traffic, weather])
        
        # 时间注意力机制
        attention = tf.keras.layers.Dense(1, activation='tanh')(concat)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(96)(attention)  # 64 + 32
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # 应用注意力
        weighted = tf.keras.layers.Multiply()([concat, attention])
        
        # 主干网络
        x = tf.keras.layers.LSTM(
            units=48,
            return_sequences=False,
            dropout=0.2
        )(weighted)
        
        # 残差连接
        traffic_pooled = tf.keras.layers.GlobalAveragePooling1D()(traffic)
        weather_pooled = tf.keras.layers.GlobalAveragePooling1D()(weather)
        residual = tf.keras.layers.Concatenate()([traffic_pooled, weather_pooled])
        
        # 输出层
        x = tf.keras.layers.Concatenate()([x, residual])
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def build_gru(self, input_shape):
        """构建增强GRU模型"""
        inputs = tf.keras.Input(shape=input_shape)
        traffic_features = inputs[:, :, :207]
        weather_features = inputs[:, :, 207:]
        
        # 天气特征处理
        weather = tf.keras.layers.BatchNormalization()(weather_features)
        weather = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            units=32,
            return_sequences=True,
            dropout=0.2
        ))(weather)
        
        # 交通特征处理
        traffic = tf.keras.layers.BatchNormalization()(traffic_features)
        traffic = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            units=64,
            return_sequences=True,
            dropout=0.2
        ))(traffic)
        
        # 特征融合
        concat = tf.keras.layers.Concatenate()([traffic, weather])
        
        # 自注意力机制
        attention = tf.keras.layers.Dense(128, activation='relu')(concat)
        attention = tf.keras.layers.Dense(1, activation='tanh')(attention)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(192)(attention)  # (64+32)*2
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        weighted = tf.keras.layers.Multiply()([concat, attention])
        
        # 主干网络
        x = tf.keras.layers.GRU(
            units=48,
            return_sequences=False,
            dropout=0.2
        )(weighted)
        
        # 残差连接
        traffic_pooled = tf.keras.layers.GlobalAveragePooling1D()(traffic)
        weather_pooled = tf.keras.layers.GlobalAveragePooling1D()(weather)
        residual = tf.keras.layers.Concatenate()([traffic_pooled, weather_pooled])
        
        x = tf.keras.layers.Concatenate()([x, residual])
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def build_cnn_lstm(self, input_shape):
        """构建增强CNN-LSTM模型"""
        inputs = tf.keras.Input(shape=input_shape)
        traffic_features = inputs[:, :, :207]
        weather_features = inputs[:, :, 207:]
        
        # 天气特征CNN处理
        weather = tf.keras.layers.BatchNormalization()(weather_features)
        weather_conv1 = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(weather)
        weather_conv2 = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(weather)
        weather = tf.keras.layers.Concatenate()([weather_conv1, weather_conv2])
        weather = tf.keras.layers.MaxPooling1D(2)(weather)
        
        # 交通特征CNN处理
        traffic = tf.keras.layers.BatchNormalization()(traffic_features)
        traffic_conv1 = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(traffic)
        traffic_conv2 = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(traffic)
        traffic = tf.keras.layers.Concatenate()([traffic_conv1, traffic_conv2])
        traffic = tf.keras.layers.MaxPooling1D(2)(traffic)
        
        # 特征融合
        concat = tf.keras.layers.Concatenate()([traffic, weather])
        
        # 双向LSTM处理
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=48,
            return_sequences=True,
            dropout=0.2
        ))(concat)
        
        # 注意力机制
        attention = tf.keras.layers.Dense(1, activation='tanh')(x)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(96)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        x = tf.keras.layers.Multiply()([x, attention])
        x = tf.keras.layers.LSTM(48)(x)
        
        # 残差连接
        traffic_pooled = tf.keras.layers.GlobalAveragePooling1D()(traffic)
        weather_pooled = tf.keras.layers.GlobalAveragePooling1D()(weather)
        residual = tf.keras.layers.Concatenate()([traffic_pooled, weather_pooled])
        
        x = tf.keras.layers.Concatenate()([x, residual])
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """训练模型
        
        Args:
            model_name: 模型名称
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            
        Returns:
            model: 训练好的模型
            history: 训练历史
        """
        # 使用静默模式
        with tf.keras.utils.CustomObjectScope({}):
            # 使用上下文管理器来抑制警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # 使用 GPU 设备
                with tf.device('/GPU:0'):
                    if model_name not in self.models:
                        if model_name == 'LSTM':
                            model = self.build_lstm(X_train.shape[1:])
                        elif model_name == 'GRU':
                            model = self.build_gru(X_train.shape[1:])
                        elif model_name == 'CNN_LSTM':
                            model = self.build_cnn_lstm(X_train.shape[1:])
                        else:
                            raise ValueError(f"Unknown model name: {model_name}")
                        
                        # 编译模型
                        model.compile(
                            optimizer=MODEL_CONFIG[model_name]['optimizer'](
                                learning_rate=MODEL_CONFIG[model_name]['learning_rate']
                            ),
                            loss=MODEL_CONFIG[model_name]['loss'],
                            metrics=MODEL_CONFIG[model_name]['metrics']
                        )
                        
                        self.models[model_name] = model
                    
                    logging.info(f"Training {model_name} for {TRAINING_CONFIG['epochs']} epochs")
                    
                    # 训练模型
                    history = self.models[model_name].fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=TRAINING_CONFIG['batch_size'],
                        epochs=TRAINING_CONFIG['epochs'],
                        verbose=1,
                        callbacks=TRAINING_CONFIG['callbacks'],
                        use_multiprocessing=True,
                        workers=4
                    )
                    
                    return self.models[model_name], history
