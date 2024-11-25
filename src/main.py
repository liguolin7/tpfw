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
from config import PROCESSED_DATA_DIR, RESULTS_DIR, RANDOM_SEED
import random
import datetime
from evaluation import plot_training_history, shap_analysis

# 固定随机种子
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def setup_logging():
    """设置日志配置"""
    # 创建必要的目录
    log_dir = 'logs'
    results_dir = 'results/experiment_outputs'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    
    # 配置处理器
    handlers = [
        logging.FileHandler(os.path.join(log_dir, 'experiment.log'), mode='w', encoding='utf-8'),
        logging.FileHandler(os.path.join(results_dir, 'experiment_output.txt'), mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers
    )
    
    # 设置所有第三方库的日志级别为 ERROR
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith(('tensorflow', 'numpy', 'matplotlib', 'keras')):
            logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # 过滤掉优化器警告
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='keras')

def create_directories():
    """创建必要的目录结构"""
    directories = [
        PROCESSED_DATA_DIR
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def train_and_evaluate_models(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    feature_names,
    experiment_name,
    results_dir
):
    """训练和评估模型"""
    logging.info(f"\n{'='*50}")
    logging.info(f"开始 {experiment_name} 实验")
    logging.info(f"{'='*50}")
    logging.info(f"数据集大小:")
    logging.info(f"训练集: {X_train.shape}")
    logging.info(f"验证集: {X_val.shape}")
    logging.info(f"测试集: {X_test.shape}")
    
    models = BaselineModels()
    results = {}
    
    # 模型配置
    model_configs = [
        ('LSTM', (models.train_lstm, lambda x: models.predict('lstm', x))),
        ('GRU', (models.train_gru, lambda x: models.predict('gru', x))),
        ('CNN_LSTM', (models.train_cnn_lstm, lambda x: models.predict('cnn_lstm', x)))
    ]
    
    for model_name, (train_func, predict_func) in model_configs:
        logging.info(f"\n{'-'*20} {model_name} 模型训练 {'-'*20}")
        
        # 重塑数据
        X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val_reshaped = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # 训练模型
        history = train_func(X_train_reshaped, y_train, X_val_reshaped, y_val)
        model = models.get_model(model_name)
        
        # 绘制学习曲线
        plot_training_history(history, model_name, results_dir)
        
        # 进行预测
        y_pred = model.predict(X_test_reshaped).flatten()
        # 评估模型
        results[model_name] = evaluate_model(y_test, y_pred, model_name)
        
        # 绘制预测结果图
        plot_predictions(y_test, y_pred, model_name, results_dir)
        
        # 执行 SHAP 分析
        X_sample = X_test_reshaped[:100]  # 示例：取前 100 个样本
        shap_analysis(model, X_sample, model_name, results_dir)
    
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
    create_directories()
    setup_logging()
    
    # 固定随机种子
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
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
    # 获取特征列表
    feature_names = data.columns.tolist()
    
    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(data)
    
    # 定义结果保存目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(RESULTS_DIR, f'{experiment_name}_results_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # 训练和评估模型
    results = train_and_evaluate_models(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_names,
        experiment_name,
        results_dir
    )
    
    return results

def save_experiment_results(baseline_results, enhanced_results, improvements):
    """保存实验结果和对比分析"""
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建结果目录，包含时间戳
    results_dir = os.path.join(RESULTS_DIR, f'comparison_{timestamp}')
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
    
    # 创建 DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存结果，文件名包含时间戳
    comparison_df.to_csv(os.path.join(results_dir, f'model_comparison_{timestamp}.csv'), index=False)
    logging.info("实验结果已保存")
    
    # 绘制对比图，传递 results_dir
    plot_model_comparison(baseline_results, enhanced_results, results_dir)

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
    """配置硬件优化"""
    try:
        # 设置环境变量
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制所有 TF 日志
        os.environ['TF_KERAS_BACKEND_LEGACY_OPTIMIZER'] = '1'
        os.environ['TF_KERAS_BACKEND_LEGACY_WARNING'] = '0'
        tf.get_logger().setLevel('ERROR')
        
        # 检测可用设备（不输出日志）
        devices = tf.config.list_physical_devices()
        
        # GPU 配置
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
                    
        # 线程优化
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        
    except Exception as e:
        logging.error(f"Hardware configuration error: {str(e)}")

if __name__ == "__main__":
    main() 