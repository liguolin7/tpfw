from data_loader import *
from data_processor import DataProcessor
from models import BaselineModels
from evaluation import *
import logging
import pandas as pd
import os
import numpy as np

def setup_logging():
    """配置日志"""
    # 创建results目录（如果不存在）
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('results/processing.log'),
            logging.StreamHandler()
        ]
    )

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """训练和评估所有模型"""
    models = BaselineModels()
    results = {}
    
    # 线性回归
    logging.info("开始训练线性回归模型...")
    models.train_linear_regression(X_train, y_train)
    y_pred = models.predict_linear_regression(X_test)
    results['LinearRegression'] = evaluate_model(y_test, y_pred, 'LinearRegression')
    plot_predictions(y_test, y_pred, 'LinearRegression')
    plot_residuals(y_test, y_pred, 'LinearRegression')
    
    # 随机森林
    logging.info("开始训练随机森林模型...")
    models.train_random_forest(X_train, y_train)
    y_pred = models.predict_random_forest(X_test)
    results['RandomForest'] = evaluate_model(y_test, y_pred, 'RandomForest')
    plot_predictions(y_test, y_pred, 'RandomForest')
    plot_residuals(y_test, y_pred, 'RandomForest')
    plot_feature_importance(models.rf_model, feature_names, 'RandomForest')
    
    # LSTM
    logging.info("开始训练LSTM模型...")
    # 重塑数据为3D格式 (samples, timesteps, features)
    X_train_3d = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_3d = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test_3d = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    models.create_lstm_model(input_shape=(1, X_train.shape[1]))
    history = models.train_lstm(X_train_3d, y_train, X_val_3d, y_val)
    y_pred = models.predict_lstm(X_test_3d)
    results['LSTM'] = evaluate_model(y_test, y_pred, 'LSTM')
    plot_predictions(y_test, y_pred, 'LSTM')
    plot_residuals(y_test, y_pred, 'LSTM')
    
    return results

def main():
    setup_logging()
    logging.info("开始数据处理...")
    
    try:
        # 加载数据
        traffic_data = load_traffic_data()
        weather_data = load_weather_data()
        
        # 数据处理
        processor = DataProcessor()
        
        # 处理交通数据
        processed_traffic = processor.process_traffic_data(traffic_data)
        
        # 处理天气数据
        processed_weather = processor.process_weather_data(weather_data)
        
        # 合并数据
        merged_data = processor.align_and_merge_data(processed_traffic, processed_weather)
        
        # 特征工程
        final_data = processor.create_features(merged_data)
        
        # 保存处理后的数据
        output_path = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')
        final_data.to_csv(output_path)
        logging.info(f"处理后的数据已保存到: {output_path}")
        
        # 划分数据集
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(final_data)
        feature_names = X_train.columns.tolist()
        
        # 训练和评估模型
        results = train_and_evaluate_models(
            X_train, X_val, X_test, 
            y_train, y_val, y_test,
            feature_names
        )
        
        # 保存总体评估结果
        pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'metrics', 'all_models_comparison.csv'))
        
    except Exception as e:
        logging.error(f"处理过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 