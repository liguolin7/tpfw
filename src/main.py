import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import importlib
import sys

# 确保每次都重新加载配置
if 'src.config' in sys.modules:
    importlib.reload(sys.modules['src.config'])
    # 重新加载相关模块
    if 'src.models' in sys.modules:
        importlib.reload(sys.modules['src.models'])

from . import config

from .data_loader import load_traffic_data, load_weather_data
from .data_processor import DataProcessor
from .models import BaselineModels, EnhancedModels
from .evaluation import evaluate_model, evaluate_models, calculate_improvements
from .visualization import DataVisualizer

# 设置环境变量以禁用所有 TensorFlow 和 CUDA 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 只使用第一个 GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 优化
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
os.environ['TF_ENABLE_RESOURCE_VARIABLES'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_DISABLE_CONTROL_FLOW_V2'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM'] = '1'

# 禁用所有 Python 警告
import warnings
warnings.filterwarnings('ignore')

# 配置 TensorFlow 日志级别
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# 禁用 TensorFlow 执行器警告
tf.debugging.disable_traceback_filtering()

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)

# 创建自定义的日志过滤器
class TensorFlowFilter(logging.Filter):
    def filter(self, record):
        return not any(msg in str(record.getMessage()).lower() for msg in [
            'executing op',
            'gradient',
            'executor',
            'custom operations',
            'numa node',
            'tf-trt',
            'tensorflow',
            'cuda',
            'gpu',
            'warning',
            'warn'
        ])

# 配置日志过滤
for handler in logging.getLogger().handlers:
    handler.addFilter(TensorFlowFilter())

# 配置TensorFlow日志
logging.getLogger('tensorflow').addFilter(TensorFlowFilter())
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# 禁用 NumPy 警告
np.seterr(all='ignore')

# 禁用 Pandas 警告
pd.options.mode.chained_assignment = None

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        os.path.join(config.RESULTS_DIR, 'figures'),
        os.path.join(config.RESULTS_DIR, 'figures', 'traffic'),
        os.path.join(config.RESULTS_DIR, 'figures', 'weather'),
        os.path.join(config.RESULTS_DIR, 'figures', 'models'),
        os.path.join(config.RESULTS_DIR, 'figures', 'comparison'),
        os.path.join(config.RESULTS_DIR, 'figures', 'analysis'),
        os.path.join(config.RESULTS_DIR, 'figures', 'training'),
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_experiment():
    """运行实验并返回结果"""
    try:
        # 获取最新配置
        training_config = config.get_training_config()
        model_config = config.get_model_config()
        
        # 添加详细的配置验证日志
        logging.info("初始配置验证:")
        logging.info(f"训练配置: epochs={training_config['epochs']}, batch_size={training_config['batch_size']}")
        logging.info(f"回调函数: {[type(cb).__name__ for cb in training_config['callbacks']]}")
        logging.info(f"优化器配置: {model_config}")
        
        data_processor = DataProcessor()
        visualizer = DataVisualizer()
        
        # 加载数据
        logging.info("开始加载数据...")
        traffic_data = load_traffic_data()
        weather_data = load_weather_data()
        logging.info(f"交通数据形状: {traffic_data.shape}, 时间范围: {traffic_data.index[0]} 到 {traffic_data.index[-1]}")
        logging.info(f"天气数据形状: {weather_data.shape}, 时间范围: {weather_data.index[0]} 到 {weather_data.index[-1]}")
        
        # 准备基准模型数据
        logging.info("准备基准模型数据...")
        X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_sequences(
            traffic_data=traffic_data,
            sequence_length=config.DATA_CONFIG['sequence_length']
        )
        logging.info(f"基准模型数据准备完成:")
        logging.info(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        
        # 准备增强模型数据
        logging.info("准备增强模型数据(包含天气特征)...")
        X_train_enhanced, y_train_enhanced, X_val_enhanced, y_val_enhanced, X_test_enhanced, y_test_enhanced = \
            data_processor.prepare_sequences(
                traffic_data=traffic_data,
                weather_data=weather_data,
                sequence_length=config.DATA_CONFIG['sequence_length']
            )
        logging.info(f"增强模型数据准备完成:")
        logging.info(f"训练集: {X_train_enhanced.shape}, 验证集: {X_val_enhanced.shape}, 测试集: {X_test_enhanced.shape}")
        
        # 训练基准模型
        logging.info("开始训练基准模型...")
        baseline_models = BaselineModels()
        baseline_metrics = {}
        baseline_predictions = {}
        
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            logging.info(f"\n开始训练基准{model_name}模型...")
            logging.info(f"配置信息: batch_size={training_config['batch_size']}, epochs={training_config['epochs']}")
            
            model, history = baseline_models.train_model(
                model_name, X_train, y_train, X_val, y_val
            )
            
            predictions = model.predict(
                X_test,
                batch_size=int(training_config['batch_size']),
                verbose=training_config['verbose']
            )
            metrics = evaluate_model(
                y_test, 
                predictions.flatten(), 
                model_name,
                model=baseline_models.models[model_name],
                feature_names=None,
                history=history
            )
            
            logging.info(f"{model_name}基准模型评估结果:")
            logging.info(f"RMSE: {metrics['RMSE']:.4f}")
            logging.info(f"MAE: {metrics['MAE']:.4f}")
            logging.info(f"R2: {metrics['R2']:.4f}")
            logging.info(f"MAPE: {metrics['MAPE']:.4f}")
            
            baseline_metrics[model_name] = metrics
            baseline_predictions[model_name] = predictions.flatten()
            
            visualizer.plot_training_history(
                history=history,
                model_name=f"{model_name}_baseline",
                save_path=os.path.join(config.RESULTS_DIR, 'figures', 'training')
            )
        
        # 训练增强模型
        logging.info("\n开始训练增强模型...")
        enhanced_models = EnhancedModels()
        enhanced_metrics = {}
        enhanced_predictions = {}
        
        # 生成特征名称
        feature_names = []
        # 生成基本特征名称
        for i in range(config.DATA_CONFIG['sequence_length']):
            feature_names.extend([
                f'traffic_t-{i}',
                f'temp_t-{i}',
                f'precip_t-{i}',
                f'wind_t-{i}',
                f'humidity_t-{i}'
            ])
        
        # 添加额外特征名称
        feature_names.extend([
            'hour_sin',
            'hour_cos',
            'day_sin',
            'day_cos',
            'is_weekend',
            'is_holiday'
        ])
        
        # 确保特征名称列表长度与输入维度匹配
        input_shape = X_train_enhanced.shape[-1]
        if len(feature_names) < input_shape:
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), input_shape)])
        elif len(feature_names) > input_shape:
            feature_names = feature_names[:input_shape]
        
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            logging.info(f"\n开始训练增强{model_name}模型(包含天气特征)...")
            logging.info(f"配置信息: batch_size={training_config['batch_size']}, epochs={training_config['epochs']}")
            
            model, history = enhanced_models.train_model(
                model_name, X_train_enhanced, y_train_enhanced, 
                X_val_enhanced, y_val_enhanced
            )
            
            predictions = model.predict(
                X_test_enhanced,
                batch_size=int(training_config['batch_size']),
                verbose=training_config['verbose']
            )
            metrics = evaluate_model(
                y_test, 
                predictions.flatten(), 
                model_name,
                model=enhanced_models.models[model_name],
                feature_names=feature_names,
                history=history
            )
            
            logging.info(f"{model_name}增强模型评估结果:")
            logging.info(f"RMSE: {metrics['RMSE']:.4f}")
            logging.info(f"MAE: {metrics['MAE']:.4f}")
            logging.info(f"R2: {metrics['R2']:.4f}")
            logging.info(f"MAPE: {metrics['MAPE']:.4f}")
            
            enhanced_metrics[model_name] = metrics
            enhanced_predictions[model_name] = predictions.flatten()
            
            visualizer.plot_training_history(
                history=history,
                model_name=f"{model_name}_enhanced",
                save_path=os.path.join(config.RESULTS_DIR, 'figures', 'training')
            )
        
        # 计算性能提升
        improvements = calculate_improvements(baseline_metrics, enhanced_metrics)
        logging.info("\n模型性能提升:")
        for model in improvements:
            logging.info(f"\n{model}模型改进:")
            for metric, value in improvements[model].items():
                logging.info(f"{metric}: {value:.2f}%")
        
        # 获取时间戳
        test_timestamps = traffic_data.index[-len(y_test):]
        
        return (baseline_metrics, enhanced_metrics, improvements,
                y_test, test_timestamps, baseline_predictions,
                enhanced_predictions, enhanced_models)
        
    except Exception as e:
        logging.error(f"实验运行过程中出现错误: {str(e)}")
        raise e

def main():
    try:
        # 创建必要的目录
        setup_directories()
        
        # 获取最新配置
        training_config = config.get_training_config()
        model_config = config.get_model_config()
        
        # 验证配置值
        logging.info(f"初始配置验证:")
        logging.info(f"epochs={training_config['epochs']}")
        logging.info(f"batch_size={training_config['batch_size']}")
        
        logging.info(f"Initial epochs value: {training_config['epochs']}")
        
        visualizer = DataVisualizer()
        
        # 运行实验取结果
        (baseline_metrics, enhanced_metrics, improvements,
         y_test, test_timestamps, baseline_predictions,
         enhanced_predictions, enhanced_models) = run_experiment()
        
        # 加载原始数据用于基础可视化
        traffic_data = load_traffic_data()
        weather_data = load_weather_data()
        
        # 生成交通数据可视化
        visualizer.plot_traffic_patterns(
            traffic_data=traffic_data,
            save_path=visualizer.subdirs['traffic']
        )
        
        # 生成天气数据可视化
        visualizer.plot_weather_analysis(
            weather_data=weather_data,
            save_path=visualizer.subdirs['weather']
        )
        
        # 生成更多数据分析可视化
        visualizer.plot_traffic_time_analysis(
            traffic_data=traffic_data,
            save_path=visualizer.subdirs['traffic']
        )

        visualizer.plot_weather_correlation_analysis(
            weather_data=weather_data,
            save_path=visualizer.subdirs['weather']
        )

        visualizer.plot_traffic_weather_relationship(
            traffic_data=traffic_data,
            weather_data=weather_data,
            save_path=visualizer.subdirs['analysis']
        )
        
        # 1. 为每个模型生成对比图
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            # 基础预测结果可视化
            visualizer.plot_prediction_vs_actual(
                y_true=y_test[:100],
                y_pred=baseline_predictions[model_name][:100],
                timestamps=test_timestamps[:100],
                model_name=f'{model_name}_baseline',
                save_path=visualizer.subdirs['models']
            )
            
            # 增强预测结果可视化
            visualizer.plot_prediction_vs_actual(
                y_true=y_test[:100],
                y_pred=enhanced_predictions[model_name][:100],
                timestamps=test_timestamps[:100],
                model_name=f'{model_name}_enhanced',
                save_path=visualizer.subdirs['models']
            )
            
            # 预测误差分布
            visualizer.plot_error_distribution(
                y_true=y_test,
                y_pred=baseline_predictions[model_name],
                model_name=f'{model_name}_baseline',
                save_path=visualizer.subdirs['models']
            )
            
            visualizer.plot_error_distribution(
                y_true=y_test,
                y_pred=enhanced_predictions[model_name],
                model_name=f'{model_name}_enhanced',
                save_path=visualizer.subdirs['models']
            )
        
        # 2. 创建所有模型的预测对比
        predictions_dict = {
            'LSTM_baseline': baseline_predictions['LSTM'],
            'LSTM_enhanced': enhanced_predictions['LSTM'],
            'GRU_baseline': baseline_predictions['GRU'],
            'GRU_enhanced': enhanced_predictions['GRU'],
            'CNN_LSTM_baseline': baseline_predictions['CNN_LSTM'],
            'CNN_LSTM_enhanced': enhanced_predictions['CNN_LSTM']
        }
        visualizer.plot_prediction_comparison(
            y_true=y_test,
            predictions_dict=predictions_dict,
            save_path=visualizer.subdirs['comparison']
        )
        
        # 3. 创建模型改进对比图
        visualizer.plot_model_improvements(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            save_path=visualizer.subdirs['comparison']
        )
        
        # 4. 创建总体性能对比表格
        visualizer.create_performance_table(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            improvements=improvements,
            save_path=visualizer.subdirs['metrics']
        )
        
        # 生成特征名称
        input_shape = enhanced_models.models['LSTM'].input_shape[-1]
        feature_names = []
        
        # 生成基本特征名称
        for i in range(config.DATA_CONFIG['sequence_length']):
            feature_names.extend([
                f'traffic_t-{i}',
                f'temp_t-{i}',
                f'precip_t-{i}',
                f'wind_t-{i}',
                f'humidity_t-{i}'
            ])
        
        # 添加额外特征名称
        feature_names.extend([
            'hour_sin',
            'hour_cos',
            'day_sin',
            'day_cos',
            'is_weekend',
            'is_holiday'
        ])
        
        # 确保特征名称列表长度与输入维度匹配
        if len(feature_names) < input_shape:
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), input_shape)])
        elif len(feature_names) > input_shape:
            feature_names = feature_names[:input_shape]
        
        # 生成特征重要性分析
        try:
            importance_df = visualizer.plot_feature_importance_analysis(
                model=enhanced_models.models['LSTM'],
                feature_names=feature_names,
                save_path=visualizer.subdirs['analysis']
            )
            if importance_df is not None:
                logging.info(f"Top 5 most important features:\n{importance_df.head()}")
        except Exception as e:
            logging.error(f"Error in feature importance analysis: {str(e)}")

        # 创建综合报告
        visualizer.create_comprehensive_report(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            weather_data=weather_data,
            save_path=visualizer.subdirs['metrics']
        )
        
        logging.info("所有模型的对比图和性能分析保存完成")
        
    except Exception as e:
        logging.error(f"实验过程中出现错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()