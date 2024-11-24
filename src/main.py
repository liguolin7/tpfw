from data_loader import *
from data_processor import DataProcessor
from models import BaselineModels
from evaluation import *
import logging
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import shap
from sklearn.model_selection import KFold

def setup_logging():
    """配置日志输出"""
    # 创建logs目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'experiment.log')
    
    # 配置日志格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    # 设置第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('shap').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names, experiment_name):
    """添加交叉验证的训练和评估"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = BaselineModels()
    results = {}
    
    # 使用tqdm创建进度条
    model_configs = [
        ('LinearRegression', (models.train_linear_regression, models.predict_linear_regression)),
        ('RandomForest', (models.train_random_forest, models.predict_random_forest)),
        ('LSTM', (models.train_lstm, models.predict_lstm))
    ]
    
    logging.info("开始模型训练和评估...")
    
    for model_name, model_func in tqdm(model_configs, desc="训练模型"):
        cv_scores = {
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        # 执行交叉验证
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 训练模型
            if model_name == 'LSTM':
                # 创建LSTM模型
                input_shape = (X_train.shape[1], 1)  # 调整输入形状
                models.create_lstm_model(input_shape)
                
                # 重塑数据以适应LSTM
                X_fold_train_reshaped = X_fold_train.values.reshape((X_fold_train.shape[0], X_fold_train.shape[1], 1))
                X_fold_val_reshaped = X_fold_val.values.reshape((X_fold_val.shape[0], X_fold_val.shape[1], 1))
                
                # 训练模型
                models.train_lstm(X_fold_train_reshaped, y_fold_train, X_fold_val_reshaped, y_fold_val)
                y_fold_pred = models.predict_lstm(X_fold_val_reshaped)
            else:
                train_func, predict_func = model_func
                train_func(X_fold_train, y_fold_train)
                y_fold_pred = predict_func(X_fold_val)
                
                # 计算验证集分数
                fold_scores = evaluate_model(y_fold_val, y_fold_pred, f"{model_name}_fold_{fold}")
                for metric in cv_scores.keys():
                    cv_scores[metric].append(fold_scores[f'test_{metric}'])
        
        # 记录交叉验证结果
        results[model_name] = {
            'cv_rmse_mean': np.mean(cv_scores['rmse']),
            'cv_rmse_std': np.std(cv_scores['rmse']),
            'cv_mae_mean': np.mean(cv_scores['mae']),
            'cv_mae_std': np.std(cv_scores['mae']),
            'cv_r2_mean': np.mean(cv_scores['r2']),
            'cv_r2_std': np.std(cv_scores['r2'])
        }
        
        # 在完整测试集上评估
        logging.info(f"开始训练{model_name}模型...")
        
        # 训练和预测
        train_func, predict_func = model_func
        
        # 对LSTM模型特殊处理
        if model_name == 'LSTM':
            # 创建LSTM模型
            input_shape = (X_train.shape[1], 1)  # 调整输入形状
            models.create_lstm_model(input_shape)
            
            # 重塑数据以适应LSTM
            X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val_reshaped = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
            X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # 训练模型
            train_func(X_train_reshaped, y_train, X_val_reshaped, y_val)
            y_pred = predict_func(X_test_reshaped)
        else:
            train_func(X_train, y_train)
            y_pred = predict_func(X_test)
        
        # 评估和可视化
        results[model_name] = evaluate_model(y_test, y_pred, model_name)
        plot_predictions(y_test, y_pred, model_name)
        plot_prediction_distribution(y_test, y_pred, model_name)
        
        # 详细的模型性能分析
        detailed_stats = detailed_model_analysis(y_test, y_pred, model_name)
        results[model_name].update({
            'detailed_stats': detailed_stats
        })
        
        # 如果是增强模型（包含天气特征）
        if 'enhanced' in experiment_name:
            try:
                weather_impact = analyze_weather_impact(
                    models.get_model(model_name),
                    X_test if model_name != 'LSTM' else X_test_reshaped,
                    y_test,
                    feature_names
                )
                results[model_name]['weather_impact'] = weather_impact
            except Exception as e:
                logging.warning(f"无法分析{model_name}的天气影响: {str(e)}")
        
        # SHAP值分析（对于支持的模型）
        try:
            if model_name != 'LSTM':  # 暂时跳过LSTM的SHAP分析
                # 限制背景数据样本数量
                background_data = shap.sample(X_test, 100)  # 只���用100个背景样本
                
                if model_name == 'LinearRegression':
                    explainer = shap.LinearExplainer(
                        models.get_model(model_name), 
                        background_data,
                        feature_names=feature_names,
                        check_additivity=False  # 禁用加性检查
                    )
                else:
                    explainer = shap.TreeExplainer(
                        models.get_model(model_name),
                        background_data,
                        feature_names=feature_names,
                        check_additivity=False  # 禁用加性检查
                    )
                
                # 只计算部分测试样本的SHAP值
                test_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
                shap_values = explainer.shap_values(test_sample)
                
                results[model_name]['shap_values'] = {
                    'values': shap_values,
                    'feature_names': feature_names
                }
        except Exception as e:
            logging.warning(f"无法为{model_name}执行SHAP分析: {str(e)}")
    
    return results

def run_experiments():
    """运行基础实验和增强实验"""
    setup_logging()
    
    logging.info("开始实验...")
    
    # 加载数据
    traffic_data = load_traffic_data()
    weather_data = load_weather_data()
    
    processor = DataProcessor()
    
    # 准备实验数据
    logging.info("数据预处理...")
    baseline_data = processor.prepare_baseline_data(traffic_data)
    enhanced_data = processor.prepare_enhanced_data(traffic_data, weather_data)
    
    # 运行实验
    logging.info("基础实验...")
    baseline_results = run_single_experiment(processor, baseline_data, 'baseline')
    
    logging.info("增强实验...")
    enhanced_results = run_single_experiment(processor, enhanced_data, 'enhanced')
    
    # 分析结果
    improvements = calculate_improvement(baseline_results, enhanced_results)
    best_model = find_best_model(improvements)
    
    logging.info(f"最佳模型: {best_model}")
    
    return baseline_results, enhanced_results, improvements

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
        feature_names,
        experiment_name
    )
    
    # 保存实验结果
    save_path = os.path.join(RESULTS_DIR, 'metrics', f'{experiment_name}_results.csv')
    pd.DataFrame(results).to_csv(save_path)
    
    return results

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
    
    # 添加现有代码的基础上
    plot_model_comparison(baseline_results, enhanced_results)

if __name__ == "__main__":
    main() 