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
from . import config  # 修改导入方式
import warnings
from sklearn.metrics import r2_score

# 配置GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 GPU 内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # 设置 GPU 选项
        tf.config.experimental.enable_tensor_float_32_execution(True)  # 启用 TF32
        tf.config.optimizer.set_jit(True)  # 启用 XLA JIT 编译
        tf.config.threading.set_inter_op_parallelism_threads(8)  # 设置线程数
        tf.config.threading.set_intra_op_parallelism_threads(8)
        
        # 设置混合精度训练
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        
# 设置日志级别
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置环境变量以优化性能
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '8'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class DataPreprocessor:
    @staticmethod
    def prepare_data(X, y):
        """准备数据，确保数据类型和格式正确"""
        # 使用 float32 而不是 float16，因为某些操作需要更高的精度
        X = tf.cast(X, tf.float32)
        y = tf.cast(y, tf.float32)
        return X, y

class BaselineModels:
    """基准模型类 - 使用简单结构"""
    def __init__(self):
        self.models = {}
        
    def build_lstm(self, input_shape):
        """构建基础LSTM模型"""
        # 获取最新配置
        model_config = config.get_model_config()
        
        model = tf.keras.Sequential([
            # 添加输入标准化层
            tf.keras.layers.BatchNormalization(input_shape=input_shape),
            
            # 第一个LSTM层
            tf.keras.layers.LSTM(
                units=model_config['LSTM']['units'][0],
                return_sequences=True,
                dropout=model_config['LSTM']['dropout'],
                recurrent_dropout=0.1,  # 添加循环dropout
                kernel_regularizer=tf.keras.regularizers.l2(
                    model_config['LSTM']['l2_regularization']
                )
            ),
            tf.keras.layers.BatchNormalization(),
            
            # 第二个LSTM层
            tf.keras.layers.LSTM(
                units=model_config['LSTM']['units'][1],
                return_sequences=False,
                dropout=model_config['LSTM']['dropout'],
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
        # 获取最新配置
        model_config = config.get_model_config()
        
        model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(input_shape=input_shape),
            
            # 双向GRU
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                units=model_config['GRU']['units'][0],
                return_sequences=True,
                dropout=model_config['GRU']['dropout'],
                recurrent_dropout=0.1
            )),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.GRU(
                units=model_config['GRU']['units'][1],
                return_sequences=False,
                dropout=model_config['GRU']['dropout']
            ),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def build_cnn_lstm(self, input_shape):
        """构建基础CNN-LSTM模型"""
        # 获取最新配置
        model_config = config.get_model_config()
        
        model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(input_shape=input_shape),
            
            # 并行卷积层
            tf.keras.layers.Conv1D(
                filters=model_config['CNN_LSTM']['cnn_filters'][0],
                kernel_size=3,
                padding='same',
                activation='relu'
            ),
            tf.keras.layers.Conv1D(
                filters=model_config['CNN_LSTM']['cnn_filters'][0],
                kernel_size=5,
                padding='same',
                activation='relu'
            ),
            
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            
            # LSTM层
            tf.keras.layers.LSTM(
                units=model_config['CNN_LSTM']['lstm_units'][0],
                return_sequences=True,
                dropout=model_config['CNN_LSTM']['dropout']
            ),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.LSTM(
                units=model_config['CNN_LSTM']['lstm_units'][1],
                return_sequences=False
            ),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """训练模型"""
        try:
            # 每次训练时重新获取最新的配置
            training_config = config.get_training_config()
            model_config = config.get_model_config()
            
            # 验证配置值
            logging.info(f"当前训练配置: epochs={training_config['epochs']}, batch_size={training_config['batch_size']}")
            
            # 预处理数据
            X_train, y_train = DataPreprocessor.prepare_data(X_train, y_train)
            X_val, y_val = DataPreprocessor.prepare_data(X_val, y_val)
            
            # 使用静默模式
            with tf.keras.utils.CustomObjectScope({}):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # 使用混合精度训练
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    
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
                            optimizer=model_config[model_name]['optimizer'](
                                learning_rate=model_config[model_name]['learning_rate']
                            ),
                            loss=model_config[model_name]['loss'],
                            metrics=['mae', 'mse', 'mape']
                        )
                        
                        self.models[model_name] = model
                    
                    logging.info(f"Training {model_name} for {training_config['epochs']} epochs")
                    
                    # 添加自定义指标回调
                    custom_metrics_callback = CustomMetricsCallback(
                        X_train, y_train, X_val, y_val,
                        batch_size=training_config['batch_size']
                    )
                    callbacks = training_config['callbacks'] + [custom_metrics_callback]
                    
                    # 使用tf.data.Dataset进行数据加载
                    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
                        .batch(training_config['batch_size'])\
                        .prefetch(tf.data.AUTOTUNE)
                    
                    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
                        .batch(training_config['batch_size'])\
                        .prefetch(tf.data.AUTOTUNE)
                    
                    # 训练模型，确保使用配置中的epochs值
                    history = self.models[model_name].fit(
                        train_dataset,
                        validation_data=val_dataset,
                        epochs=int(training_config['epochs']),  # 确保epochs是整数
                        batch_size=int(training_config['batch_size']),  # 确保batch_size是整数
                        verbose=training_config['verbose'],
                        callbacks=callbacks,
                        use_multiprocessing=False,  # 禁用多进程
                        workers=1  # 减少worker数量
                    )
                    
                    return self.models[model_name], history
                
        except Exception as e:
            logging.error(f"训练过程中出现错误: {str(e)}")
            raise e

class EnhancedModels:
    """增强模型类 - 使用更复杂的结构处理天气特征"""
    def __init__(self):
        self.models = {}
        
    def build_lstm(self, input_shape):
        """构建增强版LSTM模型"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 1. 分离交通和天气特征
        traffic_features = inputs[:, :, :207]  # 交通特征
        weather_features = inputs[:, :, 207:]  # 天气特征
        
        # 2. 交通特征处理分支
        traffic = tf.keras.layers.BatchNormalization()(traffic_features)
        traffic = tf.keras.layers.LSTM(
            units=128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2
        )(traffic)
        
        # 3. 天气特征处理分支
        weather = tf.keras.layers.BatchNormalization()(weather_features)
        weather = tf.keras.layers.LSTM(
            units=64,
            return_sequences=True,
            dropout=0.2
        )(weather)
        
        # 4. 特征融合
        concat = tf.keras.layers.Concatenate()([traffic, weather])
        
        # 5. 主干网络
        x = tf.keras.layers.LSTM(
            units=128,
            return_sequences=False,
            dropout=0.2
        )(concat)
        
        # 6. 输出层
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def build_gru(self, input_shape):
        """构建增强版GRU模型"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 1. 分离特征
        traffic_features = inputs[:, :, :207]
        weather_features = inputs[:, :, 207:]
        
        # 2. 交通特征处理
        traffic = tf.keras.layers.BatchNormalization()(traffic_features)
        traffic = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            units=128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2
        ))(traffic)
        
        # 3. 天气特征处理
        weather = tf.keras.layers.BatchNormalization()(weather_features)
        weather_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32
        )(weather, weather)
        weather = tf.keras.layers.Add()([weather, weather_attention])
        weather = tf.keras.layers.LayerNormalization()(weather)
        
        weather = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            units=64,
            return_sequences=True,
            dropout=0.2
        ))(weather)
        
        # 4. 特征融合
        concat = tf.keras.layers.Concatenate()([traffic, weather])
        
        # 5. 时间注意力
        temporal_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=32
        )(concat, concat)
        concat = tf.keras.layers.Add()([concat, temporal_attention])
        concat = tf.keras.layers.LayerNormalization()(concat)
        
        # 6. 主干网络
        x = tf.keras.layers.GRU(
            units=128,
            return_sequences=False,
            dropout=0.2
        )(concat)
        
        # 7. 残差连接
        traffic_pooled = tf.keras.layers.GlobalAveragePooling1D()(traffic)
        weather_pooled = tf.keras.layers.GlobalAveragePooling1D()(weather)
        residual = tf.keras.layers.Concatenate()([traffic_pooled, weather_pooled])
        
        # 8. 特征融合
        x = tf.keras.layers.Concatenate()([x, residual])
        
        # 9. 输出层
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def build_cnn_lstm(self, input_shape):
        """构建���强版CNN-LSTM模型"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 1. 分离特征
        traffic_features = inputs[:, :, :207]
        weather_features = inputs[:, :, 207:]
        
        # 2. 交通特征CNN处理
        traffic = tf.keras.layers.BatchNormalization()(traffic_features)
        # 多尺度CNN
        conv1 = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(traffic)
        conv2 = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(traffic)
        conv3 = tf.keras.layers.Conv1D(64, 7, padding='same', activation='relu')(traffic)
        traffic = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        traffic = tf.keras.layers.BatchNormalization()(traffic)
        
        # 3. 天气特征处理
        weather = tf.keras.layers.BatchNormalization()(weather_features)
        weather_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32
        )(weather, weather)
        weather = tf.keras.layers.Add()([weather, weather_attention])
        weather = tf.keras.layers.LayerNormalization()(weather)
        
        # CNN处理天气特征
        weather = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(weather)
        weather = tf.keras.layers.BatchNormalization()(weather)
        
        # 4. 特征融合
        concat = tf.keras.layers.Concatenate()([traffic, weather])
        
        # 5. 时间注意力
        temporal_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=32
        )(concat, concat)
        concat = tf.keras.layers.Add()([concat, temporal_attention])
        concat = tf.keras.layers.LayerNormalization()(concat)
        
        # 6. LSTM处理
        x = tf.keras.layers.LSTM(
            units=128,
            return_sequences=False,
            dropout=0.2
        )(concat)
        
        # 7. 残差连接
        traffic_pooled = tf.keras.layers.GlobalAveragePooling1D()(traffic)
        weather_pooled = tf.keras.layers.GlobalAveragePooling1D()(weather)
        residual = tf.keras.layers.Concatenate()([traffic_pooled, weather_pooled])
        
        # 8. 特征融合
        x = tf.keras.layers.Concatenate()([x, residual])
        
        # 9. 输出层
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """训练模型"""
        try:
            # 使用静默模式
            with tf.keras.utils.CustomObjectScope({}):
                # 使用上下文管理器来抑制警告
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # 每次训练时重新获取最新的配置
                    training_config = config.get_training_config()
                    model_config = config.get_model_config()
                    
                    # 验证配置值
                    logging.info(f"当前训练配置: epochs={training_config['epochs']}, batch_size={training_config['batch_size']}")
                    
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
                                optimizer=model_config[model_name]['optimizer'](
                                    learning_rate=model_config[model_name]['learning_rate']
                                ),
                                loss=model_config[model_name]['loss'],
                                metrics=['mae', 'mse', 'mape']  # 添加所有需要的指标
                            )
                            
                            self.models[model_name] = model
                        
                        logging.info(f"Before training - epochs value: {training_config['epochs']}")
                        logging.info(f"Training {model_name} for {training_config['epochs']} epochs")
                        
                        # 添加自定义指标回调
                        custom_metrics_callback = CustomMetricsCallback(
                            X_train, y_train, X_val, y_val,
                            batch_size=training_config['batch_size']
                        )
                        callbacks = training_config['callbacks'] + [custom_metrics_callback]
                        
                        # 使用tf.data.Dataset进行数据加载
                        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
                            .batch(training_config['batch_size'])\
                            .prefetch(tf.data.AUTOTUNE)
                        
                        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
                            .batch(training_config['batch_size'])\
                            .prefetch(tf.data.AUTOTUNE)
                        
                        # 训练模型，确保使用配置中的epochs值
                        history = self.models[model_name].fit(
                            train_dataset,
                            validation_data=val_dataset,
                            epochs=int(training_config['epochs']),  # 确保epochs是整数
                            batch_size=int(training_config['batch_size']),  # 确保batch_size是整数
                            verbose=training_config['verbose'],
                            callbacks=callbacks,
                            use_multiprocessing=False,  # 禁用多进程
                            workers=1  # 减少worker数量
                        )
                        
                        logging.info(f"After training - epochs value: {training_config['epochs']}")
                        return self.models[model_name], history
                        
        except Exception as e:
            logging.error(f"训练过程中出现错误: {str(e)}")
            raise e

class CustomMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size=128):
        super(CustomMetricsCallback, self).__init__()
        # 预处理数据
        self.X_train, self.y_train = DataPreprocessor.prepare_data(X_train, y_train)
        self.X_val, self.y_val = DataPreprocessor.prepare_data(X_val, y_val)
        self.batch_size = batch_size
        
    @tf.function
    def predict_batch(self, x):
        """使用@tf.function加速预测"""
        return self.model(x, training=False)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        try:
            # 分批计算预测结果
            train_pred = []
            val_pred = []
            
            # 训练集预测
            for i in range(0, len(self.X_train), self.batch_size):
                batch_x = self.X_train[i:i + self.batch_size]
                batch_pred = self.predict_batch(batch_x)
                train_pred.append(batch_pred)
            
            # 验证集预测
            for i in range(0, len(self.X_val), self.batch_size):
                batch_x = self.X_val[i:i + self.batch_size]
                batch_pred = self.predict_batch(batch_x)
                val_pred.append(batch_pred)
            
            # 合并预测结果
            train_pred = tf.concat(train_pred, axis=0)
            val_pred = tf.concat(val_pred, axis=0)
            
            # 转换为numpy数组计算指标
            train_pred_np = train_pred.numpy().flatten()
            val_pred_np = val_pred.numpy().flatten()
            y_train_np = self.y_train.numpy().flatten()
            y_val_np = self.y_val.numpy().flatten()
            
            # 计算训练集指标
            logs['rmse'] = np.sqrt(np.mean((y_train_np - train_pred_np) ** 2))
            logs['mae'] = np.mean(np.abs(y_train_np - train_pred_np))
            logs['mape'] = np.mean(np.abs((y_train_np - train_pred_np) / y_train_np)) * 100
            logs['r2'] = r2_score(y_train_np, train_pred_np)
            
            # 计算验证集指标
            logs['val_rmse'] = np.sqrt(np.mean((y_val_np - val_pred_np) ** 2))
            logs['val_mae'] = np.mean(np.abs(y_val_np - val_pred_np))
            logs['val_mape'] = np.mean(np.abs((y_val_np - val_pred_np) / y_val_np)) * 100
            logs['val_r2'] = r2_score(y_val_np, val_pred_np)
            
            # 记录当前学习率
            logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
            
        except Exception as e:
            logging.error(f"Error in CustomMetricsCallback: {str(e)}")
            raise
