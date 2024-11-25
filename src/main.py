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
import tensorflow as tf

def setup_logging():
    """配置日志输出"""
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建logs和results目录
    log_dir = 'logs'
    results_dir = 'results/experiment_outputs'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 设置日志文件路径
    log_file = os.path.join(log_dir, 'experiment.log')
    output_file = os.path.join(results_dir, 'experiment_output.txt')
    
    # 配置日志格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # 输出结果处理器
    output_handler = logging.FileHandler(output_file, mode='w', encoding='utf-8')
    output_handler.setFormatter(formatter)
    output_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, output_handler, console_handler]
    )
    
    # 设置所有第三方库的日志级别为 ERROR
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # 禁用 TensorFlow 的警告和信息输出
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names, experiment_name):
    """训练和评估模型"""
    logging.info(f"\n{'='*50}")
    logging.info(f"开始{experiment_name}实验")
    logging.info(f"{'='*50}")
    logging.info(f"数据集大小:")
    logging.info(f"训练集: {X_train.shape}")
    logging.info(f"验证集: {X_val.shape}")
    logging.info(f"测试集: {X_test.shape}")
    
    models = BaselineModels()
    results = {}
    
    # 使用自定义进度条
    model_configs = [
        ('LSTM', (models.train_lstm, models.predict_lstm)),
        ('GRU', (models.train_gru, models.predict_gru)),
        ('CNN_LSTM', (models.train_cnn_lstm, models.predict_cnn_lstm))
    ]
    
    for model_name, (train_func, predict_func) in model_configs:
        logging.info(f"\n{'-'*20} {model_name} 模型训练 {'-'*20}")
        
        # 重塑数据
        X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val_reshaped = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # 添加训练验证
        history = train_func(X_train_reshaped, y_train, X_val_reshaped, y_val)
        
        # 验证训练是否成功
        if history is None or not hasattr(history, 'history'):
            logging.error(f"{model_name} 模型训练失败")
            continue
            
        y_pred = predict_func(X_test_reshaped)
        results[model_name] = evaluate_model(y_test, y_pred, model_name)
        
    return results

def run_experiments():
    """运行对比实验"""
    try:
        # 加载数据
        traffic_data = load_traffic_data()
        weather_data = load_weather_data()
        
        # 数据预处理
        logging.info("数据预处理...")
        processor = DataProcessor()
        
        # 基础实验（只使用交通数据）
        baseline_data = processor.prepare_data(traffic_data)
        baseline_results = run_single_experiment(processor, baseline_data, 'baseline')
        
        # 增强实验（使用交通+天气数据）
        enhanced_data = processor.prepare_data(traffic_data, weather_data)
        enhanced_results = run_single_experiment(processor, enhanced_data, 'enhanced')
        
        # 计算性能提升
        improvements = calculate_improvement(baseline_results, enhanced_results)
        
        return baseline_results, enhanced_results, improvements
        
    except Exception as e:
        logging.error(f"实验过程中出现错误: {str(e)}")
        raise

def calculate_improvement(baseline_results, enhanced_results):
    """计算模型性能提升"""
    improvements = {}
    metrics = ['rmse', 'mae', 'r2', 'mape']
    
    for model in baseline_results.keys():
        improvements[model] = {}
        for metric in metrics:
            baseline = baseline_results[model][metric]
            enhanced = enhanced_results[model][metric]
            
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
    
    try:
        # 运行实验
        baseline_results, enhanced_results, improvements = run_experiments()
        
        # 保存结果并输出总结
        save_experiment_results(baseline_results, enhanced_results, improvements)
        print_experiment_summary(baseline_results, enhanced_results, improvements)
        
    except Exception as e:
        logging.error(f"实验过程中出现错误: {str(e)}")
        raise

def run_single_experiment(processor, data, experiment_name):
    """运行单个实验"""
    # 获取天气特征列表
    weather_features = [col for col in data.columns if col.startswith(('temp_', 'humidity_', 'precip_', 'wind_'))]
    
    # 在训练模型之前进行天气特征分析
    if experiment_name == 'enhanced':
        analyze_weather_features(data, weather_features)
    
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
        for metric in ['rmse', 'mae', 'r2', 'mape']:
            comparison_data.append({
                'Model': model,
                'Metric': metric,
                'Baseline': baseline_results[model][metric],
                'Enhanced': enhanced_results[model][metric],
                'Improvement(%)': improvements[model][f'{metric}_improvement']
            })
    
    # 使用concat创建DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存结果
    comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
    logging.info("实验结果已保存")
    
    # 绘制对比图
    plot_model_comparison(baseline_results, enhanced_results)

def find_best_model(improvements):
    """找出性能提升最好的模型"""
    avg_improvements = {}
    for model, metrics in improvements.items():
        avg_improvements[model] = np.mean([v for v in metrics.values()])
    
    return max(avg_improvements.items(), key=lambda x: x[1])[0]

def calculate_average_improvement(model_improvements):
    """计算平均性能提升"""
    return np.mean([v for v in model_improvements.values()])

def print_experiment_summary(baseline_results, enhanced_results, improvements):
    """打印实验总结"""
    logging.info("\n=== 实验总结 ===")
    logging.info("=" * 50)
    
    metrics = ['rmse', 'mae', 'r2', 'mape']
    models = ['LSTM', 'GRU', 'CNN_LSTM']
    
    # 只输出一次性能提升结果
    logging.info("\n性能提升结果:")
    for model in models:
        logging.info(f"\n{model} 模型:")
        for metric in metrics:
            imp = improvements[model].get(f'{metric}_improvement', 0)
            logging.info(f"{metric}_improvement: {imp:>7.2f}%")
    
    # 输出最佳模型信息
    best_model = find_best_model(improvements)
    avg_improvement = calculate_average_improvement(improvements[best_model])
    logging.info(f"\n最佳模型: {best_model}")
    logging.info(f"平均性能提升: {avg_improvement:.2f}%")

def configure_hardware():
    """配置 M1 MacBook 的硬件优化"""
    try:
        # 检测是否支持 Metal
        devices = tf.config.list_physical_devices()
        print("可用设备:", devices)
        
        # 对于 M1，使用 Metal 插件
        if len(devices) > 0:
            # 启用内存增长
            for device in devices:
                tf.config.experimental.set_memory_growth(device, True)
        
        # M1 优化的混合精度设置
        # 注意：M1 上使用 'mixed_float16' 可能会有兼容性问题
        # 建议使用默认精度设置
        
        # 设置线程优化
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        
    except Exception as e:
        print(f"硬件配置出错: {e}")

if __name__ == "__main__":
    main() 