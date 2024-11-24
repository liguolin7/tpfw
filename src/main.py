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

def run_experiments():
    """运行基础实验和增强实验"""
    # 加载数据
    traffic_data = load_traffic_data()
    weather_data = load_weather_data()
    
    processor = DataProcessor()
    
    # 准备两组实验数据
    baseline_data = processor.prepare_baseline_data(traffic_data)
    enhanced_data = processor.prepare_enhanced_data(traffic_data, weather_data)
    
    # 运行基础实验
    logging.info("开始基础实验（仅使用交通数据）...")
    baseline_results = run_single_experiment(
        processor, baseline_data, 
        experiment_name='baseline'
    )
    
    # 运行增强实验
    logging.info("开始增强实验（使用交通+天气数据）...")
    enhanced_results = run_single_experiment(
        processor, enhanced_data, 
        experiment_name='enhanced'
    )
    
    # 计算并保存性能提升
    improvement_analysis = calculate_improvement(
        baseline_results, 
        enhanced_results
    )
    
    return baseline_results, enhanced_results, improvement_analysis

def calculate_improvement(baseline_results, enhanced_results):
    """计算模型性能提升"""
    improvements = {}
    metrics = ['rmse', 'mae', 'r2']
    
    for model in baseline_results.keys():
        improvements[model] = {}
        for metric in metrics:
            baseline = baseline_results[model][f'test_{metric}']
            enhanced = enhanced_results[model][f'test_{metric}']
            
            if metric == 'r2':
                # R2提升百分比
                improvement = (enhanced - baseline) / abs(baseline) * 100
            else:
                # RMSE和MAE降低百分比
                improvement = (baseline - enhanced) / baseline * 100
                
            improvements[model][f'{metric}_improvement'] = improvement
    
    return improvements

def main():
    setup_logging()
    logging.info("开始对比实验...")
    
    try:
        # 运行两组实验
        baseline_results, enhanced_results, improvements = run_experiments()
        
        # 保存结果
        save_experiment_results(
            baseline_results, 
            enhanced_results, 
            improvements
        )
        
        # 输出最佳模型
        find_best_model(improvements)
        
    except Exception as e:
        logging.error(f"实验过程中出现错误: {str(e)}")
        raise

def run_single_experiment(processor, data, experiment_name):
    """运行单个实验（基础或增强）"""
    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(data)
    feature_names = X_train.columns.tolist()
    
    # 训练和评估模型
    results = train_and_evaluate_models(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_names
    )
    
    # 保存实验结果
    save_path = os.path.join(RESULTS_DIR, 'metrics', f'{experiment_name}_results.csv')
    pd.DataFrame(results).to_csv(save_path)
    
    return results  # 直接返回结果字典

def save_experiment_results(baseline_results, enhanced_results, improvements):
    """保存实验结果和对比分析"""
    # 创建结果目录
    results_dir = os.path.join(RESULTS_DIR, 'comparison')
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备对比结果数据
    comparison_data = []
    
    for model in improvements.keys():
        for metric in ['rmse', 'mae', 'r2']:
            comparison_data.append({
                'Model': model,
                'Metric': metric,
                'Baseline': baseline_results[model][f'test_{metric}'],
                'Enhanced': enhanced_results[model][f'test_{metric}'],
                'Improvement(%)': improvements[model][f'{metric}_improvement']
            })
    
    # 使用concat创建DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存结果
    comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
    logging.info("实验结果已保存")

if __name__ == "__main__":
    main() 