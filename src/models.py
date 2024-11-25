from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import logging
from config import RANDOM_STATE
from tqdm import tqdm

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
        """训练带有正则化的线性回归模型"""
        logging.info("训练线性回归模型...")
        self.lr_model = Ridge(
            alpha=1.0,  # 正则化强度
            random_state=42
        )
        
        try:
            with tqdm(total=1, desc="LR Training") as pbar:
                self.lr_model.fit(X_train, y_train)
                score = self.lr_model.score(X_train, y_train)
                pbar.update(1)
            
            logging.info(f"线性回归训练集 R2 分数: {score:.4f}")
        except Exception as e:
            logging.error(f"线性回归训练失败: {str(e)}")
            raise
        
    def train_random_forest(self, X_train, y_train):
        """训练随机森林模型"""
        logging.info("训练随机森林模型...")
        
        # 限制树的数量和深度，避免过度消耗内存和计算资源
        self.rf_model = RandomForestRegressor(
            n_estimators=50,  # 减少树的数量
            max_depth=10,     # 限制树的深度
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=4  # 限制并行数，避免占用过多CPU
        )
        
        try:
            with tqdm(total=1, desc="RF Training") as pbar:
                self.rf_model.fit(X_train, y_train)
                score = self.rf_model.score(X_train, y_train)
                pbar.update(1)
            
            logging.info(f"随机森林训练集 R2 分数: {score:.4f}")
        except Exception as e:
            logging.error(f"随机森林训练失败: {str(e)}")
            raise
        
    def create_lstm_model(self, input_shape):
        """创建改进的LSTM模型"""
        self.lstm_model = Sequential([
            # 第一个LSTM层
            LSTM(100, input_shape=input_shape, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            # 第二个LSTM层
            LSTM(50, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # 全连接层
            Dense(25, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            # 输出层
            Dense(1)
        ])
        
        # 编译模型
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # 使用Huber损失函数，对异常值更稳健
            metrics=['mae']
        )
        
        logging.info(f"创建LSTM模型，输入形状: {input_shape}")
        self.lstm_model.summary()
        
    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
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
        
    def get_model(self, model_name):
        """获取指定名称的模型对象"""
        model_map = {
            'LinearRegression': self.lr_model,
            'RandomForest': self.rf_model,
            'LSTM': self.lstm_model
        }
        return model_map.get(model_name)