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
from . import config  # modify the import method
import warnings
from sklearn.metrics import r2_score

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set GPU options
        tf.config.experimental.enable_tensor_float_32_execution(True)  # Enable TF32
        tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
        tf.config.threading.set_inter_op_parallelism_threads(8)  # Set the number of threads
        tf.config.threading.set_intra_op_parallelism_threads(8)
        
        # Set mixed precision training
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        
# Set log level
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set environment variables for performance optimization
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '8'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class DataPreprocessor:
    @staticmethod
    def prepare_data(X, y):
        """Prepare data, ensure correct data types and formats"""
        # Use float32 instead of float16, as some operations require higher precision
        X = tf.cast(X, tf.float32)
        y = tf.cast(y, tf.float32)
        return X, y

class BaselineModels:
    """Baseline model class - use simple structures"""
    def __init__(self):
        self.models = {}
        
    def build_lstm(self, input_shape):
        """Build basic LSTM model"""
        # Get the latest configuration
        model_config = config.get_model_config()
        
        model = tf.keras.Sequential([
            # Add input normalization layer
            tf.keras.layers.BatchNormalization(input_shape=input_shape),
            
            # First LSTM layer
            tf.keras.layers.LSTM(
                units=model_config['LSTM']['units'][0],
                return_sequences=True,
                dropout=model_config['LSTM']['dropout'],
                recurrent_dropout=0.1,  # Add recurrent dropout
                kernel_regularizer=tf.keras.regularizers.l2(
                    model_config['LSTM']['l2_regularization']
                )
            ),
            tf.keras.layers.BatchNormalization(),
            
            # Second LSTM layer
            tf.keras.layers.LSTM(
                units=model_config['LSTM']['units'][1],
                return_sequences=False,
                dropout=model_config['LSTM']['dropout'],
                recurrent_dropout=0.1
            ),
            
            # Fully connected layer
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def build_gru(self, input_shape):
        """Build basic GRU model"""
        # Get the latest configuration
        model_config = config.get_model_config()
        
        model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(input_shape=input_shape),
            
            # Bidirectional GRU
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
        """Build basic CNN-LSTM model"""
        # Get the latest configuration
        model_config = config.get_model_config()
        
        model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(input_shape=input_shape),
            
            # Parallel convolutional layers
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
            
            # LSTM layer
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
        """Train the model"""
        try:
            # Get the latest configuration every time you train
            training_config = config.get_training_config()
            model_config = config.get_model_config()
            
            # Validate configuration values
            logging.info(f"Current training configuration: epochs={training_config['epochs']}, batch_size={training_config['batch_size']}")
            
            # Preprocess data
            X_train, y_train = DataPreprocessor.prepare_data(X_train, y_train)
            X_val, y_val = DataPreprocessor.prepare_data(X_val, y_val)
            
            # Use silent mode
            with tf.keras.utils.CustomObjectScope({}):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Use mixed precision training
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
                        
                        # Compile the model
                        model.compile(
                            optimizer=model_config[model_name]['optimizer'](
                                learning_rate=model_config[model_name]['learning_rate']
                            ),
                            loss=model_config[model_name]['loss'],
                            metrics=['mae', 'mse', 'mape']
                        )
                        
                        self.models[model_name] = model
                    
                    logging.info(f"Training {model_name} for {training_config['epochs']} epochs")
                    
                    # Add custom metrics callback
                    custom_metrics_callback = CustomMetricsCallback(
                        X_train, y_train, X_val, y_val,
                        batch_size=training_config['batch_size']
                    )
                    callbacks = training_config['callbacks'] + [custom_metrics_callback]
                    
                    # Use tf.data.Dataset for data loading
                    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
                        .batch(training_config['batch_size'])\
                        .prefetch(tf.data.AUTOTUNE)
                    
                    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
                        .batch(training_config['batch_size'])\
                        .prefetch(tf.data.AUTOTUNE)
                    
                    # Train the model, make sure to use the epochs value in the configuration
                    history = self.models[model_name].fit(
                        train_dataset,
                        validation_data=val_dataset,
                        epochs=int(training_config['epochs']),  # Make sure epochs is an integer
                        batch_size=int(training_config['batch_size']),  # Make sure batch_size is an integer
                        verbose=training_config['verbose'],
                        callbacks=callbacks,
                        use_multiprocessing=False,  # Disable multiprocessing
                        workers=1  # Reduce the number of workers
                    )
                    
                    return self.models[model_name], history
                
        except Exception as e:
            logging.error(f"Error occurred during training: {str(e)}")
            raise e

class EnhancedModels:
    """Enhanced model class - use more complex structures to handle weather features"""
    def __init__(self):
        self.models = {}
        
    def build_lstm(self, input_shape):
        """Build enhanced LSTM model"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 1. Separate traffic and weather features
        traffic_features = inputs[:, :, :207]  # Traffic features
        weather_features = inputs[:, :, 207:]  # Weather features
        
        # 2. Traffic feature processing branch
        traffic = tf.keras.layers.BatchNormalization()(traffic_features)
        traffic = tf.keras.layers.LSTM(
            units=128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2
        )(traffic)
        
        # 3. Weather feature processing branch
        weather = tf.keras.layers.BatchNormalization()(weather_features)
        weather = tf.keras.layers.LSTM(
            units=64,
            return_sequences=True,
            dropout=0.2
        )(weather)
        
        # 4. Feature fusion
        concat = tf.keras.layers.Concatenate()([traffic, weather])
        
        # 5. Main network
        x = tf.keras.layers.LSTM(
            units=128,
            return_sequences=False,
            dropout=0.2
        )(concat)
        
        # 6. Output layer
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def build_gru(self, input_shape):
        """Build enhanced GRU model"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 1. Separate features
        traffic_features = inputs[:, :, :207]
        weather_features = inputs[:, :, 207:]
        
        # 2. Traffic feature processing
        traffic = tf.keras.layers.BatchNormalization()(traffic_features)
        traffic = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            units=128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2
        ))(traffic)
        
        # 3. Weather feature processing
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
        
        # 4. Feature fusion
        concat = tf.keras.layers.Concatenate()([traffic, weather])
        
        # 5. Temporal attention
        temporal_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=32
        )(concat, concat)
        concat = tf.keras.layers.Add()([concat, temporal_attention])
        concat = tf.keras.layers.LayerNormalization()(concat)
        
        # 6. Main network
        x = tf.keras.layers.GRU(
            units=128,
            return_sequences=False,
            dropout=0.2
        )(concat)
        
        # 7. Residual connection
        traffic_pooled = tf.keras.layers.GlobalAveragePooling1D()(traffic)
        weather_pooled = tf.keras.layers.GlobalAveragePooling1D()(weather)
        residual = tf.keras.layers.Concatenate()([traffic_pooled, weather_pooled])
        
        # 8. Feature fusion
        x = tf.keras.layers.Concatenate()([x, residual])
        
        # 9. Output layer
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def build_cnn_lstm(self, input_shape):
        """Build enhanced CNN-LSTM model"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 1. Separate features
        traffic_features = inputs[:, :, :207]
        weather_features = inputs[:, :, 207:]
        
        # 2. Traffic feature CNN processing
        traffic = tf.keras.layers.BatchNormalization()(traffic_features)
        # Multi-scale CNN
        conv1 = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(traffic)
        conv2 = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(traffic)
        conv3 = tf.keras.layers.Conv1D(64, 7, padding='same', activation='relu')(traffic)
        traffic = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        traffic = tf.keras.layers.BatchNormalization()(traffic)
        
        # 3. Weather feature processing
        weather = tf.keras.layers.BatchNormalization()(weather_features)
        weather_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32
        )(weather, weather)
        weather = tf.keras.layers.Add()([weather, weather_attention])
        weather = tf.keras.layers.LayerNormalization()(weather)
        
        # CNN processing weather features
        weather = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(weather)
        weather = tf.keras.layers.BatchNormalization()(weather)
        
        # 4. Feature fusion
        concat = tf.keras.layers.Concatenate()([traffic, weather])
        
        # 5. Temporal attention
        temporal_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=32
        )(concat, concat)
        concat = tf.keras.layers.Add()([concat, temporal_attention])
        concat = tf.keras.layers.LayerNormalization()(concat)
        
        # 6. LSTM processing
        x = tf.keras.layers.LSTM(
            units=128,
            return_sequences=False,
            dropout=0.2
        )(concat)
        
        # 7. Residual connection
        traffic_pooled = tf.keras.layers.GlobalAveragePooling1D()(traffic)
        weather_pooled = tf.keras.layers.GlobalAveragePooling1D()(weather)
        residual = tf.keras.layers.Concatenate()([traffic_pooled, weather_pooled])
        
        # 8. Feature fusion
        x = tf.keras.layers.Concatenate()([x, residual])
        
        # 9. Output layer
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """Train the model"""
        try:
            # Use silent mode
            with tf.keras.utils.CustomObjectScope({}):
                # Use context manager to suppress warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Get the latest configuration every time you train
                    training_config = config.get_training_config()
                    model_config = config.get_model_config()
                    
                    # Validate configuration values
                    logging.info(f"Current training configuration: epochs={training_config['epochs']}, batch_size={training_config['batch_size']}")
                    
                    # Use GPU device
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
                            
                            # Compile the model
                            model.compile(
                                optimizer=model_config[model_name]['optimizer'](
                                    learning_rate=model_config[model_name]['learning_rate']
                                ),
                                loss=model_config[model_name]['loss'],
                                metrics=['mae', 'mse', 'mape']  # Add all required metrics
                            )
                            
                            self.models[model_name] = model
                        
                        logging.info(f"Before training - epochs value: {training_config['epochs']}")
                        logging.info(f"Training {model_name} for {training_config['epochs']} epochs")
                        
                        # Add custom metrics callback
                        custom_metrics_callback = CustomMetricsCallback(
                            X_train, y_train, X_val, y_val,
                            batch_size=training_config['batch_size']
                        )
                        callbacks = training_config['callbacks'] + [custom_metrics_callback]
                        
                        # Use tf.data.Dataset for data loading
                        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
                            .batch(training_config['batch_size'])\
                            .prefetch(tf.data.AUTOTUNE)
                        
                        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
                            .batch(training_config['batch_size'])\
                            .prefetch(tf.data.AUTOTUNE)
                        
                        # Train the model, make sure to use the epochs value in the configuration
                        history = self.models[model_name].fit(
                            train_dataset,
                            validation_data=val_dataset,
                            epochs=int(training_config['epochs']),  # Make sure epochs is an integer
                            batch_size=int(training_config['batch_size']),  # Make sure batch_size is an integer
                            verbose=training_config['verbose'],
                            callbacks=callbacks,
                            use_multiprocessing=False,  # Disable multiprocessing
                            workers=1  # Reduce the number of workers
                        )
                        
                        logging.info(f"After training - epochs value: {training_config['epochs']}")
                        return self.models[model_name], history
                        
        except Exception as e:
            logging.error(f"Error occurred during training: {str(e)}")
            raise e

class CustomMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size=128):
        super(CustomMetricsCallback, self).__init__()
        # Preprocess the data
        self.X_train, self.y_train = DataPreprocessor.prepare_data(X_train, y_train)
        self.X_val, self.y_val = DataPreprocessor.prepare_data(X_val, y_val)
        self.batch_size = batch_size
        
    @tf.function
    def predict_batch(self, x):
        """Use @tf.function to accelerate prediction"""
        return self.model(x, training=False)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        try:
            # Calculate predictions in batches
            train_pred = []
            val_pred = []
            
            # Predict on training set
            for i in range(0, len(self.X_train), self.batch_size):
                batch_x = self.X_train[i:i + self.batch_size]
                batch_pred = self.predict_batch(batch_x)
                train_pred.append(batch_pred)
            
            # Predict on validation set
            for i in range(0, len(self.X_val), self.batch_size):
                batch_x = self.X_val[i:i + self.batch_size]
                batch_pred = self.predict_batch(batch_x)
                val_pred.append(batch_pred)
            
            # Concatenate the predictions
            train_pred = tf.concat(train_pred, axis=0)
            val_pred = tf.concat(val_pred, axis=0)
            
            # Convert to numpy arrays for metric calculation
            train_pred_np = train_pred.numpy().flatten()
            val_pred_np = val_pred.numpy().flatten()
            y_train_np = self.y_train.numpy().flatten()
            y_val_np = self.y_val.numpy().flatten()
            
            # Calculate metrics on training set
            logs['rmse'] = np.sqrt(np.mean((y_train_np - train_pred_np) ** 2))
            logs['mae'] = np.mean(np.abs(y_train_np - train_pred_np))
            logs['mape'] = np.mean(np.abs((y_train_np - train_pred_np) / y_train_np)) * 100
            logs['r2'] = r2_score(y_train_np, train_pred_np)
            
            # Calculate metrics on validation set
            logs['val_rmse'] = np.sqrt(np.mean((y_val_np - val_pred_np) ** 2))
            logs['val_mae'] = np.mean(np.abs(y_val_np - val_pred_np))
            logs['val_mape'] = np.mean(np.abs((y_val_np - val_pred_np) / y_val_np)) * 100
            logs['val_r2'] = r2_score(y_val_np, val_pred_np)
            
            # Record the current learning rate
            logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
            
        except Exception as e:
            logging.error(f"Error in CustomMetricsCallback: {str(e)}")
            raise
