import os
import logging
import pandas as pd
from datetime import datetime
import numpy as np
from data_loader import load_traffic_data, load_weather_data
from data_processor import DataProcessor
from models import BaselineModels, EnhancedModels
from evaluation import evaluate_models, calculate_improvements
from visualization import DataVisualizer
from config import RESULTS_DIR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_results_directory():
    """创建结果目录并返回路径"""
    # 使用当前时间戳创建目录名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建基础实验结果目录
    baseline_dir = os.path.join(RESULTS_DIR, f'baseline_results_{timestamp}')
    os.makedirs(baseline_dir, exist_ok=True)
    
    # 创建增强实验结果目录
    enhanced_dir = os.path.join(RESULTS_DIR, f'enhanced_results_{timestamp}')
    os.makedirs(enhanced_dir, exist_ok=True)
    
    # 创建图表目录
    figures_dir = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    return baseline_dir, enhanced_dir, figures_dir

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        os.path.join(RESULTS_DIR, 'experiment_outputs'),
        os.path.join(RESULTS_DIR, 'figures'),
        os.path.join(RESULTS_DIR, 'figures', 'traffic'),
        os.path.join(RESULTS_DIR, 'figures', 'weather'),
        os.path.join(RESULTS_DIR, 'figures', 'models'),
        os.path.join(RESULTS_DIR, 'figures', 'comparison')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_experiment():
    """运行实验并生成可视化"""
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 创建目录结构
    setup_directories()
    baseline_dir, enhanced_dir, figures_dir = create_results_directory()
    
    # 创建可视化器
    visualizer = DataVisualizer()
    
    # 创建数据处理器
    data_processor = DataProcessor()
    
    # 加载数据
    traffic_data = load_traffic_data()
    weather_data = load_weather_data()
    
    # 生成初始数据分析可视化
    logging.info("生成数据分析可视化...")
    visualizer.plot_traffic_patterns(
        traffic_data,
        save_path=os.path.join(figures_dir, 'traffic')
    )
    
    visualizer.plot_weather_impact(
        weather_data,
        traffic_data,
        save_path=os.path.join(figures_dir, 'weather')
    )
    
    # 数据预处理
    logging.info("数据预处理...")
    # 使用DataProcessor类的方法处理数据
    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_sequences(traffic_data, weather_data)
    
    # 基准实验
    logging.info("\n==================================================")
    logging.info("开始 baseline 实验")
    logging.info("==================================================")
    
    baseline_models = BaselineModels()
    baseline_histories = {}
    baseline_predictions = {}
    
    for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
        logging.info(f"\n训练 {model_name} 模型...")
        model, history = baseline_models.train_model(model_name, X_train, y_train, X_val, y_val)
        baseline_histories[model_name] = history
        predictions = model.predict(X_test)
        baseline_predictions[model_name] = predictions.flatten()  # 确保预测结果是1D数组
        
        # 立即评估并打印结果
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        logging.info(f"\n{model_name} 模型评估结果:")
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")
    
    # 增强实验
    logging.info("\n==================================================")
    logging.info("开始 enhanced 实验")
    logging.info("==================================================")
    
    enhanced_models = EnhancedModels()
    enhanced_histories = {}
    enhanced_predictions = {}
    
    for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
        logging.info(f"\n训练 {model_name} 模型...")
        model, history = enhanced_models.train_model(model_name, X_train, y_train, X_val, y_val)
        enhanced_histories[model_name] = history
        predictions = model.predict(X_test)
        enhanced_predictions[model_name] = predictions.flatten()  # 确保预测结果是1D数组
        
        # 立即评估并打印结果
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        logging.info(f"\n{model_name} 模型评估结果:")
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")
    
    # 生成模型训练过程可视化
    logging.info("\n生成模型训练过程可视化...")
    visualizer.plot_model_performance(
        baseline_histories,
        save_path=os.path.join(figures_dir, 'models', 'baseline')
    )
    
    visualizer.plot_model_performance(
        enhanced_histories,
        save_path=os.path.join(figures_dir, 'models', 'enhanced')
    )
    
    # 确保y_test也是1D数组
    y_test_flat = y_test.flatten() if isinstance(y_test, np.ndarray) else y_test.values
    
    # 生成预测结果对比可视化
    logging.info("\n生成预测结果对比可视化...")
    visualizer.plot_prediction_comparison(
        y_test_flat,
        baseline_predictions,
        save_path=os.path.join(figures_dir, 'comparison', 'baseline')
    )
    
    visualizer.plot_prediction_comparison(
        y_test_flat,
        enhanced_predictions,
        save_path=os.path.join(figures_dir, 'comparison', 'enhanced')
    )
    
    # 评估模型并生成性能对比可视化
    baseline_metrics = evaluate_models(baseline_predictions, y_test_flat)
    enhanced_metrics = evaluate_models(enhanced_predictions, y_test_flat)
    
    # 计算性能提升
    improvements = calculate_improvements(baseline_metrics, enhanced_metrics)
    
    visualizer.plot_model_comparison(
        baseline_metrics,
        save_path=os.path.join(figures_dir, 'comparison', 'baseline_metrics')
    )
    
    visualizer.plot_model_comparison(
        enhanced_metrics,
        save_path=os.path.join(figures_dir, 'comparison', 'enhanced_metrics')
    )
    
    # 在评估完所有模型后，添加基准模型和增强模型的对比可视化
    visualizer.plot_model_improvements(
        baseline_metrics=baseline_metrics,
        enhanced_metrics=enhanced_metrics,
        save_path='results/figures'
    )
    
    logging.info("\n实验完成，所有可视化内容已保存到results/figures目录")
    
    return baseline_metrics, enhanced_metrics, improvements

def main():
    try:
        baseline_metrics, enhanced_metrics, improvements = run_experiment()
        logging.info("实验成功完成")
    except Exception as e:
        logging.error(f"实验过程中出现错误: {str(e)}")
        raise 
    
    # 在评估完所有模型后，添加新的可视化内容
    
    # 1. 预测结果可视化（以CNN-LSTM为例）
    visualizer.plot_prediction_vs_actual(
        y_true=y_test[:100],  # 展示前100个时间步的预测结果
        y_pred=best_model_predictions[:100],
        timestamps=test_timestamps[:100],
        model_name='CNN_LSTM',
        save_path='results/figures'
    )
    
    # 2. 特征重要性分析
    if hasattr(best_model, 'feature_importance_'):
        visualizer.plot_feature_importance(
            feature_importance=best_model.feature_importance_,
            feature_names=feature_names,
            save_path='results/figures'
        )
    
    # 3. 预测误差分布
    visualizer.plot_error_distribution(
        y_true=y_test,
        y_pred=best_model_predictions,
        model_name='CNN_LSTM',
        save_path='results/figures'
    )
    
    # 4. 创建演示总结
    visualizer.create_presentation_summary(
        baseline_metrics=baseline_metrics,
        enhanced_metrics=enhanced_metrics,
        save_path='results/figures'
    )

if __name__ == '__main__':
    main()